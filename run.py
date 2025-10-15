"""
모델 시연용 코드 (단색 테마 + 중앙정렬 헤더)
"""

import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch.utils.data import DataLoader
import plotly.graph_objects as go

# ===== 페이지 설정 =====
st.set_page_config(
    page_title="암 환자 위험도 예측 시스템",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== 세션 상태 초기화 =====
st.session_state.setdefault("pred_time", None)
st.session_state.setdefault("time_val", None)
st.session_state.setdefault("event_val", None)

# ===== 비교차트 함수 =====
def comparison_chart(pred_time, time_val):
    pred_point = float(pred_time)
    real_point = float(time_val)
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=0, y1=max(95, pred_point, real_point) + 10,
                  line=dict(color="lightgrey", width=1), layer="below")
    fig.add_trace(go.Scatter(x=[0], y=[pred_point], mode="text",
                             text=[f"<b>{int(pred_point)}</b>▶"],
                             textposition="middle left",
                             name="예측 생존 기간",
                             textfont=dict(size=21, color="#7fbdff"),
                             hoverinfo="none"))
    fig.add_trace(go.Scatter(x=[0], y=[real_point], mode="text",
                             text=[f"◀<b>{int(real_point)}</b>"],
                             textposition="middle right",
                             name="실제 생존 기간",
                             textfont=dict(size=21, color="#ff7e7e"),
                             hoverinfo="none"))
    fig.update_layout(
        title=dict(text="<b>실제 생존 기간vs예측 생존 기간</b>", x=0.5, font=dict(size=16)),
        xaxis=dict(visible=False),
        yaxis=dict(title="생존 기간(단위 : 3개월)",
                   range=[-3, max(pred_point, real_point) + 15],
                   showgrid=True, gridcolor="lightgrey"),
        height=550, plot_bgcolor="white", margin=dict(t=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="black")))
    return fig

# ===== 모듈 임포트 =====
import modules.DataAnalysis as DataAnalysis
import modules.ModelAnalysis_kmj as ModelAnalysis
import modules.DataModify as DataModify
from modules.DataSelect import DataPreprocessing
import modules.Models as Models

# ===== 데이터/모델 준비 =====
test_file = ["./data/test dataset_fixed.csv"]
test_dataset = DataModify.CancerDataset(
    target_column="event", time_column="time", file_paths=test_file
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

input_dim, hidden_size, time_bins, num_events = 17, (128, 64), 91, 4
device = torch.device("cpu")

input_params_path = "./parameters/deephit_model_2D_CNN.pth"
encoding_map = DataPreprocessing.load_category()
str_encoding_map = ModelAnalysis.clean_encoding_map(encoding_map, convert_values_to_str=True)
dp = DataPreprocessing(categories=str_encoding_map)

model = Models.DeepHitSurvWithSEBlockAnd2DCNN(input_dim, hidden_size, time_bins, num_events)
model.load_state_dict(torch.load(input_params_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

# ===== CSS (단색 버전 + 중앙정렬 타이틀 흰색) =====
st.markdown("""
<style>
:root{
  --c0:#EEF5FF;
  --c1:#B4D4FF;
  --c2:#86B6F6;
  --c3:#176B87;
}
html, body, .stApp { background:var(--c0); color:#0f172a; }


/* Tabs */
.stTabs [data-baseweb="tab-list"]{
  gap:12px; background:#fff; padding:.8rem; border-radius:14px; border-bottom:none;
}
.stTabs [data-baseweb="tab"]{
  background:#fff; color:#245; border:1px solid var(--c1);
  border-radius:10px; padding:10px 18px; font-weight:600;
  box-shadow:0 2px 8px rgba(23,107,135,.06);
}
.stTabs [aria-selected="true"]{
  background: var(--c2); color:#fff !important; border:1px solid var(--c2);
  box-shadow:0 6px 16px rgba(23,107,135,.25);
}
.stTabs [aria-selected="true"] *{ color:#fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{ display:none !important; }

/* Buttons */
.stButton > button{
  color:#fff !important; font-weight:800 !important;
  background: var(--c2) !important; border:none !important; border-radius:12px !important;
  padding:.7rem 1.2rem !important; font-size:1.04rem !important;
  box-shadow:0 8px 20px rgba(23,107,135,.22) !important;
  transition: transform .2s ease, box-shadow .2s ease !important;
}
.stButton > button:hover{
  transform: translateY(-1px) !important;
  box-shadow:0 12px 28px rgba(23,107,135,.28) !important;
  background:#5f9eec !important;
}

/* Card */
.card{
  background:#fff; border:1px solid var(--c1); border-radius:16px;
  padding:16px 20px; margin:12px 0;
  box-shadow:0 6px 22px rgba(23,107,135,.08);
}

/* Plotly container */
.plotly-container{
  border:1px dashed var(--c1); border-radius:14px; padding:10px; background:#fff;
}

/* Metric card */
.stat{
  border:1px solid var(--c1); border-radius:14px; padding:12px 14px;
  background:#fff; box-shadow:0 3px 12px rgba(23,107,135,.08);
}
.stat .label{ color:#335; font-size:.9rem; }
.stat .value{ color:var(--c3); font-size:1.35rem; font-weight:800; }

/* Tab titles 중앙정렬 + 흰색 폰트 */
.tab-title{
  display:flex; align-items:center; justify-content:center; gap:10px;
  margin:14px 0 18px; font-weight:900; color:#fff; font-size:1.2rem;
  background:var(--c2); padding:8px 18px; border-radius:999px;
  box-shadow:0 4px 10px rgba(23,107,135,.2);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ▼ Streamlit selectbox(드롭다운) 흰색 테마 강제 적용 */

/* 입력창(닫힌 상태) 컨테이너 */
.stSelectbox div[role="combobox"]{
  background:#ffffff !important;
  border:2px solid var(--c1) !important;   /* B4D4FF */
  border-radius:10px !important;
  color:#0f172a !important;
}

/* 내부 텍스트/placeholder */
.stSelectbox div[role="combobox"] *{
  color:#0f172a !important;
}

/* 포커스/호버 시 보더 컬러 */
.stSelectbox div[role="combobox"]:hover,
.stSelectbox div[role="combobox"]:focus-within{
  border-color:var(--c2) !important;       /* 86B6F6 */
}

/* 펼친 옵션 리스트 배경/보더 */
.stSelectbox [role="listbox"]{
  background:#ffffff !important;
  border:1px solid var(--c1) !important;
  border-radius:10px !important;
  box-shadow:0 8px 20px rgba(23,107,135,.12) !important;
}

/* 각 옵션 색상 */
.stSelectbox [role="option"]{
  background:#ffffff !important;
  color:#0f172a !important;
}

/* 하이라이트된(hover/선택) 옵션 */
.stSelectbox [role="option"][aria-selected="true"],
.stSelectbox [role="option"]:hover{
  background:rgba(134,182,246,.15) !important;  /* var(--c2) 15% */
  color:#0f172a !important;
}

/* 입력창 내부 인풋(검색 가능한 select일 때) */
.stSelectbox div[role="combobox"] input{
  background:#ffffff !important;
  color:#0f172a !important;
}

/* 아이콘 영역도 흰 배경 유지 */
.stSelectbox div[data-baseweb="select"] > div{
  background:#ffffff !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.hero-pretty {
  position: relative;
  margin: 12px 0 28px 0;
  padding: 26px 28px;
  border-radius: 20px;
  background: #ffffff;
  border: 3px solid var(--c1);
  box-shadow: 0 10px 25px rgba(23,107,135,.18);
  text-align: center;
}

.hero-pretty::before {
  content: "";
  position: absolute;
  top: 0; left: 50%;
  transform: translateX(-50%);
  width: 40%;
  height: 4px;
  border-radius: 3px;
  background: var(--c2);
  box-shadow: 0 3px 10px rgba(23,107,135,.3);
}

.hero-pretty h1 {
  font-size: 2.4rem;
  color: var(--c3);
  font-weight: 900;
  letter-spacing: 0.5px;
  margin-bottom: 6px;
}

.hero-pretty h1 span {
  font-size: 2.6rem;
  vertical-align: middle;
}

.hero-pretty p {
  font-size: 1.05rem;
  color: #3b4b5a;
  margin-top: 4px;
  font-weight: 500;
}

.hero-badge {
  display: inline-block;
  margin-top: 10px;
  background: var(--c2);
  color: white;
  padding: 4px 14px;
  font-size: 0.85rem;
  border-radius: 999px;
  font-weight: 700;
  box-shadow: 0 3px 10px rgba(134,182,246,.4);
}
</style>

<div class="hero-pretty">
  <h1><span>💉</span> 암 환자 고위험군 선별 및 예측 시스템</h1>
  <p>의료 데이터 기반 맞춤형 생존 예측 모델 데모</p>
</div>
""", unsafe_allow_html=True)



# ===== 유틸 함수 =====
def ui_card(md): st.markdown(f'<div class="card">{md}</div>', unsafe_allow_html=True)
def ui_stat(label, value):
    st.markdown(f'<div class="stat"><div class="label">{label}</div><div class="value">{value}</div></div>',
                unsafe_allow_html=True)
def apply_plotly_theme(fig):
    fig.update_layout(font=dict(family="Inter, Pretendard", size=14, color="#0f172a"),
                      paper_bgcolor="#fff", plot_bgcolor="#fff",
                      legend=dict(bgcolor="rgba(238,245,255,.7)",
                                  bordercolor="#B4D4FF", borderwidth=1))
    return fig



# ===== 탭 구성 =====
df = pd.read_csv("./data/categories_select.csv")
tab1, tab2 = st.tabs(["환자 정보 입력", "샘플 예측"])

# ------------------- 탭1 -------------------
with tab1:
    st.markdown('<div class="tab-title">환자 정보 입력 및 예측</div>', unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 2])
    with col_left:
        selected_values = {}
        if "Primary Site" in df.columns and "Primary Site - labeled" in df.columns:
            mapping = dict(zip(df["Primary Site - labeled"], df["Primary Site"]))
            labels = sorted(df["Primary Site - labeled"].dropna().unique().tolist())
            sel = st.selectbox("🎯 Primary Site 선택", labels)
            selected_values["Primary Site - labeled"], selected_values["Primary Site"] = sel, mapping[sel]
        for col in df.columns:
            if col in ["Primary Site", "Primary Site - labeled"]: continue
            vals = sorted(df[col].dropna().unique().tolist())
            if vals:
                emoji = {"Age":"👤","Sex":"⚥","Race":"🌍","Stage":"📊","Grade":"📈",
                         "Tumor Size":"📏","Surgery":"🔪","Radiation":"☢️","Chemotherapy":"💊"}.get(col,"📝")
                selected_values[col] = st.selectbox(f"{emoji} {col} 선택", vals)
        st.markdown("---")
        predict_button = st.button("예측 실행 🔮", key="main_predict", use_container_width=True)

        with st.expander("🔑 특성 번역 사전"):
            ui_card("""
            <table style='width:100%; border-collapse:collapse; font-size:0.95rem;'>
                <tr><td><b>Sex</b></td><td>성별</td></tr>
                <tr><td><b>Age recode with &lt;1 year olds and 90+</b></td><td>연령대</td></tr>
                <tr><td><b>Year of diagnosis</b></td><td>진단 연도</td></tr>
                <tr><td><b>Year of follow-up recode</b></td><td>추적 연도</td></tr>
                <tr><td><b>Race recode (W, B, AI, API)</b></td><td>인종 재코드</td></tr>
                <tr><td><b>Site recode ICD-O-3/WHO 2008</b></td><td>암 부위 재코드</td></tr>
                <tr><td><b>Primary Site</b></td><td>원발 부위</td></tr>
                <tr><td><b>Primary Site - labeled</b></td><td>원발 부위 라벨</td></tr>
                <tr><td><b>Derived Summary Grade 2018 (2018+)</b></td><td>요약 등급 2018</td></tr>
                <tr><td><b>Laterality</b></td><td>좌우 구분</td></tr>
                <tr><td><b>EOD Schema ID Recode (2010+)</b></td><td>EOD 스키마 재코드</td></tr>
                <tr><td><b>Combined Summary Stage with Expanded Regional Codes (2004+)</b></td><td>SEER 요약 병기(확장)</td></tr>
                <tr><td><b>RX Summ--Surg Prim Site (1998+)</b></td><td>수술 코드</td></tr>
                <tr><td><b>RX Summ--Scope Reg LN Sur (2003+)</b></td><td>림프절 절제 범위</td></tr>
                <tr><td><b>RX Summ--Surg Oth Reg/Dis (2003+)</b></td><td>기타 수술</td></tr>
                <tr><td><b>Sequence number</b></td><td>순서 번호</td></tr>
                <tr><td><b>Median household income inflation adj to 2023</b></td><td>가구 소득(2023 물가보정)</td></tr>
                <tr><td><b>Number of Cores Positive Recode (2010+)</b></td><td>양성 코어 수</td></tr>
                <tr><td><b>Number of Cores Examined Recode (2010+)</b></td><td>검사 코어 수</td></tr>
                <tr><td><b>EOD Primary Tumor Recode (2018+)</b></td><td>EOD 원발 종양</td></tr>
                <tr><td><b>PRCDA 2020</b></td><td>PRCDA 2020</td></tr>
                <tr><td><b>Survival months</b></td><td>생존 개월</td></tr>
                <tr><td><b>Survival months_bin_3m</b></td><td>생존 개월(3개월 구간)</td></tr>
                <tr><td><b>target_label</b></td><td>타깃 라벨</td></tr>
                <tr><td><b>Vital status recode (study cutoff used)__enc</b></td><td>생존 상태(인코딩)</td></tr>
            </table>
            """)


    with col_right:
        if not predict_button:
            ui_card("👈 왼쪽에서 환자 정보를 입력하고 예측 실행 버튼을 클릭하세요!")

# ==================== 탭1: 예측 실행 로직 (버튼 눌렀을 때) ====================
if "predict_button" in locals() and predict_button:
    with tab1:
        # 오른쪽 패널에 결과 출력
        with col_right:
            with st.spinner("AI가 예측을 수행 중입니다..."):
                # 🔹 입력 템플릿 로드 (첫 행을 복사해서 사용)
                base_df = pd.read_csv("./data/Suicide.csv")
                input_df = base_df.iloc[[0]].copy()

                # 🔹 왼쪽에서 선택한 값들로 덮어쓰기 (문자형으로 통일)
                for col, val in selected_values.items():
                    if col in input_df.columns and val is not None:
                        input_df.at[0, col] = str(val)

                # 🔹 인코딩 & 예측
                _ = dp.run(input_df)  # 필요하면 반환값 사용
                result_df = ModelAnalysis.predict_event_probabilities(
                    input_df=input_df, dp=dp, model=model, device=device
                )

                st.success("✅ 예측이 완료되었습니다!")
                st.markdown("---")
                st.markdown("### 😇 상세 생존 분석")

                # 🔹 단일 예측 시각화 (모델 내부 함수 사용)
                ModelAnalysis.visualize_single_prediction(
                    input_df=input_df, dp=dp, model=model, device=device
                )


# ------------------- 탭2 -------------------
with tab2:
    st.markdown('<div class="tab-title">샘플 데이터 예측</div>', unsafe_allow_html=True)
    sui_df = pd.read_csv("./data/Suicide.csv")
    col1, col2 = st.columns([1, 2])
    with col1:
        options = {"생존": -1, "암 관련 사망": 0, "합병증 관련 사망": 1, "기타 질환 사망": 2, "자살/자해": 3}
        selected_event_name = st.selectbox("🎯예측할 사건 라벨 선택", list(options.keys()))
        selected_event_label = options[selected_event_name]
        st.markdown("---")
        sample_predict_button = st.button("샘플 예측 실행 🎲", key="sample_predict", use_container_width=True)
        st.markdown("---")
        with st.expander("📖 샘플 예측이란?"):
            ui_card("테스트 데이터셋에서 선택한 사건 라벨에 해당하는 샘플을 무작위로 선택해 예측을 수행합니다.")
    with col2:
        if not sample_predict_button:
            ui_card("👈 왼쪽에서 사건 라벨을 선택하고 샘플 예측 실행 버튼을 클릭하세요!")
        else:
            import random
            indices = [i for i, (_, _, event) in enumerate(test_dataset) if event == selected_event_label]
            if not indices:
                st.warning("선택한 사건 라벨에 해당하는 샘플이 없습니다.")
            else:
                idx = random.choice(indices)
                x, st.session_state.time_val, event_val = test_dataset[idx]
                sample_input = x.unsqueeze(0)
                with torch.no_grad():
                    df_input = pd.DataFrame(sample_input.numpy())
                    ModelAnalysis.predict_event_probabilities(df_input, model=model, device=device)
                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    ui_stat("🕐 실제 관측 시간", f"{st.session_state.time_val} 개월")
                with col2_2:
                    event_map = {-1: "생존", 0: "암 관련 사망", 1: "합병증 사망", 2: "기타 질환", 3: "자살/자해"}
                    ui_stat("📋 실제 발생 사건", event_map.get(event_val, f"사건 {event_val}"))
                st.markdown("---")
                st.markdown("### 📊 샘플 상세 분석")
                st.session_state.pred_time = ModelAnalysis.visualize_single_prediction(
                    input_df=df_input, model=model, device=device,
                    time_column="time", target_column="event",
                    event_weights=[3.0, 5.0, 5.0, 10.0])
                with col1:
                    if st.session_state.pred_time is not None and st.session_state.time_val is not None:
                        fig = apply_plotly_theme(comparison_chart(st.session_state.pred_time, st.session_state.time_val))
                        st.markdown('<div class="plotly-container">', unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        ui_card("아직 비교차트가 없습니다. 아래 샘플 예측 실행을 눌러 생성하세요.")
