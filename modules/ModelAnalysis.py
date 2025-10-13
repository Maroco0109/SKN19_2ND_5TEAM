"""

모델 분석 시각화 모듈

- fit된 모델을 인자로 받아서 수행

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import torch

from modules.Models import compute_risk_score_sigmoid

def predict_event_probabilities(
    input_df: pd.DataFrame,
    model,
    device: torch.device,
    dp=None,
    time_column: str = 'Survival months_bin_3m',
    target_column: str = 'target_label'
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
    if dp is not None :
        processed_df = dp.run(input_df)
    else :
        processed_df = input_df

    drop_cols = [col for col in [time_column, target_column] if col in processed_df.columns]
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
        names=['Event', 'Time']
    )

    result_df = pd.DataFrame(cif_array.flatten()[None, :], columns=columns)

    return result_df

def visualize_single_prediction(input_df, model, device,
                                time_column='Survival months_bin_3m',
                                target_column='target_label',
                                dp=None,
                                event_weights=None,
                                time_lambda=0.05):
    """
    단일 입력 데이터(1행 DataFrame)에 대해 PMF와 CIF를 시각화
    마지막 시간 bin(dummy)은 제거됨
    """

    if dp is not None :
        processed_df = dp.run(input_df)
    else :
        processed_df = input_df

    drop_cols = [col for col in [time_column, target_column] if col in processed_df.columns]
    features_df = processed_df.drop(columns=drop_cols)

    x = torch.tensor(features_df.values.astype(float), dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        _, pmf, cif = model(x)  # (1, num_events, time_bins)

    pmf = pmf[:, :, :-1]
    cif = cif[:, :, :-1]
    _, num_events, time_bins = cif.shape
    time_points = list(range(time_bins))

    fig_pmf, ax_pmf = plt.subplots(figsize=(8, 4))
    for k in range(num_events):
        ax_pmf.plot(time_points, pmf[0, k].cpu().numpy().flatten(), label=f'Event {k+1}')
    ax_pmf.set_xlabel('Time bins')
    ax_pmf.set_ylabel('Probability (PMF)')
    ax_pmf.set_title('PMF (Probability Mass Function)')
    ax_pmf.legend()
    ax_pmf.grid(True)
    ax_pmf.set_xlim(0, 90)
    ax_pmf.set_ylim(0, 1)
    st.pyplot(fig_pmf)

    fig_cif, ax_cif = plt.subplots(figsize=(8, 4))
    for k in range(num_events):
        ax_cif.plot(time_points, cif[0, k].cpu().numpy().flatten(), label=f'Event {k+1}')
    ax_cif.set_xlabel('Time bins')
    ax_cif.set_ylabel('Cumulative Probability (CIF)')
    ax_cif.set_title('CIF (Cumulative Incidence Function)')
    ax_cif.legend()
    ax_cif.grid(True)
    ax_cif.set_xlim(0, 90)
    ax_cif.set_ylim(0, 1)
    st.pyplot(fig_cif)

    pmf_np = pmf[0].cpu().numpy()  # (num_events, time_bins)
    survival_probs = []
    surv = 1.0

    for t in range(time_bins):
        surv *= (1 - np.sum(pmf_np[:, t]))  # 모든 사건이 발생하지 않을 확률
        survival_probs.append(surv)

    fig_surv, ax_surv = plt.subplots(figsize=(8, 4))
    ax_surv.plot(time_points, survival_probs, color='black', linewidth=2)
    ax_surv.set_xlabel('Time bins')
    ax_surv.set_ylabel('Survival Probability S(t)')
    ax_surv.set_title('Survival Curve (No Event Occurrence Probability)')
    ax_surv.grid(True)
    ax_surv.set_xlim(0, 90)
    ax_surv.set_ylim(0, 1)
    st.pyplot(fig_surv)

    risk_score = compute_risk_score_sigmoid(pmf, time_lambda=time_lambda, event_weights=event_weights)
    st.subheader("⚠️ 위험 점수 (Risk Score)")
    st.write(f"{risk_score.item():.2f} / 100")

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
    time_index = torch.arange(1, pmf.shape[2] + 1, device=pmf.device).float()  # [1, 2, ..., T]
    expected_time = (survival * time_index).sum(dim=1)  # (B,)

    return {
        'survival': survival,
        'risk_score': risk_score,
        'expected_time': expected_time
    }

def clean_encoding_map(encoding_map, convert_values_to_str=True):
    cleaned_map = {}
    for col, mapping in encoding_map.items():
        # mapping이 dict인지 확인
        if not isinstance(mapping, dict):
            print(f"Warning: '{col}'의 mapping이 dict가 아니어서 건너뜀")
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


# 예시
def show_model_graph(model, x, y, e, cols) :
    pass