"""

모델 분석 시각화 모듈

- fit된 모델을 인자로 받아서 수행

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import torch

from modules.Models import compute_risk_score_sigmoid


def show_risk_level_with_emoji(risk_score):
    """
    위험도 점수에 따라 이모티콘과 메시지 표시

    Args:
        risk_score: 0~1 사이의 위험도 점수
    """
    if risk_score < 0.3:
        emoji = "😊"
        level = "낮음"
        color = "#28a745"
        bg_color = "#d4edda"
    elif risk_score < 0.6:
        emoji = "😐"
        level = "보통"
        color = "#ffc107"
        bg_color = "#fff3cd"
    elif risk_score < 0.8:
        emoji = "😰"
        level = "높음"
        color = "#fd7e14"
        bg_color = "#ffe5d0"
    else:
        emoji = "😱"
        level = "매우 높음"
        color = "#dc3545"
        bg_color = "#f8d7da"

    # 전체 위험도 표시
    st.markdown(
        f"""
        <div style="text-align: center; padding: 30px; margin: 20px 0;
                    background-color: {bg_color}; 
                    border-radius: 15px; border: 3px solid {color};">
            <div style="font-size: 60px; margin-bottom: 10px;">{emoji}</div>
            <h2 style="color: {color}; margin: 10px 0;">종합 위험도: {level}</h2>
            <h3 style="color: {color};">위험도 점수: {risk_score:.1%}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )


def predict_event_probabilities(
    input_df: pd.DataFrame,
    model,
    device: torch.device,
    dp=None,
    time_column: str = "Survival months_bin_3m",
    target_column: str = "target_label",
) -> pd.DataFrame:
    """
    1행 입력 데이터를 받아 전처리 후, 모델로 CIF 전체 예측 (마지막 시간 bin 제거)

    Args:
        input_df: 1행짜리 DataFrame
        dp: DataPreprocessing 인스턴스
        model: 학습된 PyTorch 모델
        device: 'cpu' 또는 'cuda'
        time_column: 시간 컬럼명
        target_column: 타겟 컬럼명

    Returns:
        pd.DataFrame: 사건별 × 시간별 CIF (마지막 시간 bin 제거)
    """
    if dp is not None:
        processed_df = dp.run(input_df)
    else:
        processed_df = input_df

    drop_cols = [
        col for col in [time_column, target_column] if col in processed_df.columns
    ]
    features_df = processed_df.drop(columns=drop_cols)

    x = torch.tensor(features_df.values.astype(float), dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        _, pmf, cif = model(x)  # cif: (1, num_events, time_bins)

    cif = cif[:, :, :-1]  # shape: (1, num_events, time_bins-1)

    num_events, num_time = cif.shape[1], cif.shape[2]
    cif_array = cif[0].cpu().numpy()  # (num_events, time_bins-1)

    time_points = [f"Time_{t}" for t in range(num_time)]
    columns = pd.MultiIndex.from_product(
        [[f"Event_{i}" for i in range(num_events)], time_points],
        names=["Event", "Time"],
    )

    result_df = pd.DataFrame(cif_array.flatten()[None, :], columns=columns)

    return result_df


def visualize_single_prediction(
    input_df,
    model,
    device,
    time_column="Survival months_bin_3m",
    target_column="target_label",
    dp=None,
    event_weights=None,
    time_lambda=0.05,
):
    if dp is not None:
        processed_df = dp.run(input_df)
    else:
        processed_df = input_df

    drop_cols = [
        col for col in [time_column, target_column] if col in processed_df.columns
    ]
    features_df = processed_df.drop(columns=drop_cols)

    x = torch.tensor(features_df.values.astype(float), dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        _, pmf, cif = model(x)

    pmf = pmf[:, :, :-1]
    cif = cif[:, :, :-1]
    _, num_events, time_bins = cif.shape
    time_points = list(range(time_bins))

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57"]
    event_names = ["암 관련 사망", "합병증 관련 사망", "기타 질환 사망", "자살/자해"]

    risk_score = compute_risk_score_sigmoid(
        pmf, time_lambda=time_lambda, event_weights=event_weights
    )
    normalized_risk = risk_score.item() / 100.0
    show_risk_level_with_emoji(normalized_risk)

    # ===== PMF (마커 제거, 선 원래 굵기) =====
    fig_pmf = go.Figure()
    for k in range(num_events):
        fig_pmf.add_trace(
            go.Scatter(
                x=time_points,
                y=pmf[0, k].cpu().numpy().flatten(),
                mode="lines",  # ✅ 마커 제거
                name=event_names[k] if k < len(event_names) else f"Event {k}",
                line=dict(color=colors[k % len(colors)], width=2),  # ✅ 원래 굵기 복원
                hovertemplate="<b>%{fullData.name}</b><br>시간: %{x}개월<br>확률: %{y:.4f}<extra></extra>",
            )
        )
    fig_pmf.update_layout(
        title=dict(text="📈 PMF (Probability Mass Function) - 사건별 발생 확률", x=0.5),
        xaxis_title="시간 (3개월 단위)",
        yaxis_title="발생 확률",
        yaxis=dict(range=[0, 0.2]),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_pmf, use_container_width=True)

    # ===== CIF (마커 제거, 선 원래 굵기) =====
    fig_cif = go.Figure()
    for k in range(num_events):
        fig_cif.add_trace(
            go.Scatter(
                x=time_points,
                y=cif[0, k].cpu().numpy().flatten(),
                mode="lines",  # ✅ 마커 제거
                name=event_names[k] if k < len(event_names) else f"Event {k}",
                line=dict(color=colors[k % len(colors)], width=2),
                fill="tonexty" if k > 0 else "tozeroy",
                fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(colors[k % len(colors)])) + [0.1])}",
                hovertemplate="<b>%{fullData.name}</b><br>시간: %{x}개월<br>누적 확률: %{y:.4f}<extra></extra>",
            )
        )
    fig_cif.update_layout(
        title=dict(text="📈 CIF (Cumulative Incidence Function) - 누적 발생 확률", x=0.5),
        xaxis_title="시간 (3개월 단위)",
        yaxis_title="누적 발생 확률",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_cif, use_container_width=True)

    # ===== 생존 곡선 (마커 제거, 선 원래 굵기, 기준선 원래 두께) =====
    cif_np = cif[0].cpu().numpy()
    num_events, time_bins = cif_np.shape
    survival_probs = []
    pred_time = None
    for t in range(time_bins):
        surv = 1 - np.sum(cif_np[:, t])
        survival_probs.append(surv)
        if surv <= 0.9 and pred_time is None:
            pred_time = t
    if pred_time is None:
        pred_time = time_bins - 1

    fig_surv = go.Figure()
    fig_surv.add_trace(
        go.Scatter(
            x=time_points,
            y=survival_probs,
            mode="lines",  # ✅ 마커 제거
            name="생존 확률",
            line=dict(color="#2c3e50", width=2),  # ✅ 원래 굵기 복원
            fill="tozeroy",
            fillcolor="rgba(44,62,80,0.1)",
            hovertemplate="<b>생존 확률</b><br>시간: %{x}개월<br>확률: %{y:.4f}<extra></extra>",
        )
    )

    if pred_time is not None and pred_time < time_bins - 1:
        fig_surv.add_vline(
            x=pred_time,
            line_dash="dash",
            line_color="red",
            line_width=2,  # ✅ 기준선 원래 두께
            annotation_text=f"90% 생존 시점: {pred_time}개월",
            annotation_position="top",
        )
        fig_surv.add_hline(
            y=0.9,
            line_dash="dash",
            line_color="red",
            line_width=2,  # ✅ 기준선 원래 두께
            annotation_text="90% 생존 확률",
            annotation_position="left",
        )

    fig_surv.update_layout(
        title=dict(text="📈 생존 곡선 (Survival Curve) - 사건 미발생 확률", x=0.5),
        xaxis_title="시간 (3개월 단위)",
        yaxis_title="생존 확률 S(t)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    st.plotly_chart(fig_surv, use_container_width=True)

    return pred_time



def dataset_to_dataframe(ds):
    data_list = []
    for x, t, e in ds:  # Dataset이 (x, t, e) 반환
        # x가 Tensor라면 numpy로 변환
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(t, torch.Tensor):
            t = t.item()  # 단일 값일 경우
        if isinstance(e, torch.Tensor):
            e = e.item()  # 단일 값일 경우

        row = list(x) + [t, e]  # features + time + event
        data_list.append(row)

    # 컬럼 이름 생성
    num_features = len(data_list[0]) - 2
    columns = [f"feature_{i}" for i in range(num_features)] + ["time", "event"]

    df = pd.DataFrame(data_list, columns=columns)
    return df


def compute_survival_metrics(pmf: torch.Tensor):
    """
    DeepHit 모델 출력(PMF)로부터 주요 생존 지표 계산

    Args:
        pmf (torch.Tensor): 사건별 시간대 확률 분포 (B, E, T)
            - B: batch_size
            - E: num_events
            - T: time_bins

    Returns:
        dict: {
            'survival': (B, T) 생존확률,
            'risk_score': (B,) 사건발생 위험도,
            'expected_time': (B,) 기대 생존시간
        }
    """
    # ----- CIF (누적 사건 확률) -----
    cif = torch.cumsum(pmf, dim=2)  # (B, E, T)

    # cif: (B, E, T) - cumulative incidence function
    # pmf: (B, E, T) - 사건 발생 확률

    # ----- 생존 확률 (독립 사건 가정) -----
    survival = torch.prod(1 - cif, dim=1)  # (B, T)

    # ----- 위험도 (전체 사건 발생 확률 합) -----
    risk_score = pmf.sum(dim=(1, 2))  # (B,)

    # ----- 기대 생존 시간 -----
    time_index = torch.arange(
        1, pmf.shape[2] + 1, device=pmf.device
    ).float()  # [1, 2, ..., T]
    expected_time = (survival * time_index).sum(dim=1)  # (B,)

    return {
        "survival": survival,
        "risk_score": risk_score,
        "expected_time": expected_time,
    }


def clean_encoding_map(encoding_map, convert_values_to_str=True):
    cleaned_map = {}
    for col, mapping in encoding_map.items():
        # mapping이 dict인지 확인
        if not isinstance(mapping, dict):
            continue

        new_mapping = {}
        for k, v in mapping.items():
            # np.int64, np.float64 등 제거
            if hasattr(k, "item"):
                k = k.item()
            if convert_values_to_str:
                v = str(v)
            elif hasattr(v, "item"):
                v = v.item()
            new_mapping[k] = v
        cleaned_map[col] = new_mapping
    return cleaned_map
