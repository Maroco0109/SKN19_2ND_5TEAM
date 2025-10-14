"""
모델 시연용 코드

최적의 성능을 내는 모델을 수행
- 예측을 실행할 데이터 파일은 실행 시 인자로 전달
  혹은 실행 후 파일 이름을 전달하여 수행
- 예측된 데이터를 csv 파일로 저장
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from torch.utils.data import DataLoader

import torch

# 🎨 페이지 설정 - 와이드 레이아웃과 아이콘
st.set_page_config(
    page_title="암 환자 위험도 예측 시스템",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

import modules.DataAnalysis as DataAnalysis
import modules.ModelAnalysis_kmj as ModelAnalysis
import modules.DataModify as DataModify
from modules.DataSelect import DataPreprocessing

import modules.Models as Models

# Dataset 로드
test_file = ["./data/test dataset_fixed.csv"]
test_dataset = DataModify.CancerDataset(
    target_column="event", time_column="time", file_paths=test_file
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

input_dim = 17  # input dimension : data의 feature의 개수
hidden_size = (128, 64)  # 1번째, 2번째 hidden layer의 size
time_bins = 91  # 3개월 단위로 time을 split하여 각 구간으로 삼음 -> 최대 270개월 + 그 후
num_events = 4  # 사건의 개수

input_params_path = "./parameters/deephit_model_2D_CNN.pth"
device = torch.device("cpu")

encoding_map = DataPreprocessing.load_category()
print(type(encoding_map))

# 예시: 모든 값도 문자열로 변환하려면 convert_values_to_str=True
str_encoding_map = ModelAnalysis.clean_encoding_map(
    encoding_map, convert_values_to_str=True
)

dp = DataPreprocessing(categories=str_encoding_map)

# 모델 정의 (학습할 때 사용한 모델 클래스)
model = Models.DeepHitSurvWithSEBlockAnd2DCNN(
    input_dim,
    hidden_size,
    time_bins,
    num_events,
)  # 사건 수 맞게 설정
model.load_state_dict(
    torch.load(input_params_path, map_location=device, weights_only=True)
)
model.to(device)
model.eval()  # 평가 모드

# 🎨 커스텀 CSS 스타일 - 의료 시스템용 화이트 & 블루 테마
st.markdown(
    """
<style>
    /* ========== 기본 배경 및 텍스트 ========== */
    .stApp {
        background: #ffffff;
        color: #1e293b;  /* 기본 텍스트 진한 회색 */ 
    }
    
    .main {
        background: #ffffff;
        color: #1e293b;
    }
    
    /* ========== 모든 텍스트 기본 색상 (흰 배경용) ========== */
    p, span, div, label, h1, h2, h3, h4, h5, h6 {
        color: #1e293b;
    }
    
    /* ========== Streamlit 기본 텍스트 ========== */
    .stMarkdown {
        color: #1e293b;
    }
    
    /* ========== 제목 스타일 ========== */
    .main-title {
        color: #1e3a8a !important;  /* 진한 블루 */
        text-align: center;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 2rem;
        padding: 1rem;
        border-bottom: 3px solid #3b82f6;  /* 블루 포인트 */
    }
    
    /* ========== 섹션 헤더 ========== */
    .section-header {
        background: #f8fafc;  /* 매우 연한 블루 그레이 */
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(59, 130, 246, 0.1);
        margin: 1.5rem 0; 
    }
    
    .section-header h2 {
        color: #1e40af !important;  /* 블루 */
        margin: 0;
        font-weight: 600;
    }
    
    /* ========== 카드 스타일 ========== */
    .info-card {
        background: #ffffff;
        padding: 1.8rem;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.08);
        margin: 1rem 0;
        border: 1px solid #e0e7ff;  /* 연한 블루 보더 */
    }
    
    /* ========== 사이드바 ========== */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: #f8fafc;  /* 화이트 계열 */
        border-right: 2px solid #e0e7ff;
    }
    
    /* 사이드바는 연한 배경이므로 진한 글자 */
    [data-testid="stSidebar"] * {
        color: #1e293b;
    }
    
    /* ========== 버튼 스타일 (블루 배경이므로 흰 글자) ========== */
    .stButton > button {
        background: #3b82f6 !important;  /* 블루 포인트 */
        border: none;
        border-radius: 10px;
        color: #ffffff !important;  /* 흰 글자 */
        font-weight: 600;
        padding: 0.6rem 2.5rem;
        font-size: 1.05rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.25);
        width: 100%;
    }
    
    /* 버튼 내부 텍스트도 강제로 흰색 */
    .stButton > button p,
    .stButton > button span,
    .stButton > button div {
        color: #ffffff !important;
    }
    
    .stButton > button:hover {
        background: #2563eb !important;  /* 진한 블루 */
        color: #ffffff !important;  /* 흰 글자 유지 */
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.35);
    }
    
    .stButton > button:active {
        background: #1d4ed8 !important;
        color: #ffffff !important;  /* 흰 글자 유지 */
        transform: translateY(0);
    }
    
    /* ========== 탭 스타일 ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: #f8fafc;
        padding: 0.5rem;    
        padding-top: 2rem;
        border-radius: 12px;
        border-bottom: none !important;  /* 밑줄 제거 */
    }
    
    /* 선택되지 않은 탭 (흰 배경이므로 진한 글자) */
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 12px 24px;
        color: #64748b !important;  /* 회색 글자 */
        font-weight: 500;
        border: 1px solid #e2e8f0;
        border-bottom: none !important;  /* 빨간 밑줄 제거 */
    }
    
    /* 선택된 탭 (블루 배경이므로 흰 글자) */
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;  /* 블루 포인트 */
        color: #ffffff !important;  /* 흰 글자 */
        border: 1px solid #3b82f6;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.25);
        border-bottom: none !important;  /* 빨간 밑줄 제거 */
    }
    
    /* 선택된 탭 내의 모든 요소도 흰 글자 */
    .stTabs [aria-selected="true"] * {
        color: #ffffff !important;
    }
    
    /* 탭 하단 빨간 표시선 완전 제거 */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent !important;
        display: none !important;
    }
    
    .stTabs [data-baseweb="tab-border"] {
        background-color: transparent !important;
        display: none !important;
    }
    
    /* ========== 셀렉트 박스 (항상 흰 배경에 블루 글자) ========== */
    .stSelectbox label {
        color: #1e40af;  /* 블루 라벨 */
        font-weight: 600;
    }
    
    /* 셀렉트 박스 기본 상태 - 항상 흰색 */
    .stSelectbox > div > div {
        background-color: #ffffff !important;  /* 항상 흰 배경 */
        border: 2px solid #cbd5e1;
        border-radius: 8px;
        color: #1e40af !important;  /* 블루 텍스트 */
        font-weight: 500;
    }
    
    /* 셀렉트 박스 호버 상태 - 흰색 유지 */
    .stSelectbox > div > div:hover {
        background-color: #ffffff !important;  /* 호버시에도 흰 배경 */
        border-color: #3b82f6;
    }
    
    /* 셀렉트 박스 내부 요소들 */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #ffffff !important;
    }
    
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #1e40af !important;  /* 블루 텍스트 */
    }
    
    /* 셀렉트 박스 옵션 */
    .stSelectbox [role="option"] {
        color: #1e40af !important;
    }
    
    /* 포커스 상태 - 흰색 유지 */
    .stSelectbox > div > div:focus-within {
        background-color: #ffffff !important;  /* 포커스시에도 흰 배경 */
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
    }
    
    /* 드롭다운 메뉴 */
    [data-baseweb="popover"] {
        background-color: #ffffff !important;
    }
    
    [data-baseweb="popover"] li {
        background-color: #ffffff !important;
        color: #1e40af !important;
    }
    
    [data-baseweb="popover"] li:hover {
        background-color: #eff6ff !important;  /* 호버시만 연한 블루 */
        color: #2563eb !important;
    }
    
    /* ========== 메트릭 카드 (흰 배경이므로 진한 글자) ========== */
    [data-testid="stMetricValue"] {
        color: #1e40af;  /* 블루 */
        font-weight: 600;
    }
    
    [data-testid="stMetricLabel"] {
        color: #475569;
    }
    
    /* ========== 알림 박스 (연한 배경이므로 진한 글자) ========== */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        background-color: #eff6ff;
        color: #1e40af;
    }
    
    /* ========== 인포 박스 (연한 배경이므로 진한 글자) ========== */
    .stInfo {
        background-color: #eff6ff !important;  /* 연한 블루 */
        border-left-color: #3b82f6 !important;
        color: #1e40af !important;
    }
    
    .stSuccess {
        background-color: #f0fdf4 !important;  /* 연한 그린 */
        border-left-color: #22c55e !important;
        color: #166534 !important;
    }
    
    .stWarning {
        background-color: #fef3c7 !important;  /* 연한 옐로우 */
        border-left-color: #f59e0b !important;
        color: #92400e !important;
    }
    
    /* ========== 구분선 ========== */
    hr {
        border-color: #e0e7ff;
        margin: 2rem 0;
    }
    
    /* ========== Expander (흰 배경, 클릭 시 검정) ========== */
    .streamlit-expanderHeader {
        background-color: #ffffff !important;
        border: 1px solid #e0e7ff;
        border-radius: 8px;
        color: #1e40af !important;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #f8fafc !important;
        border-color: #3b82f6;
    }
    
    /* Expander 열렸을 때 (검정 배경) */
    details[open] > .streamlit-expanderHeader {
        background-color: #1e293b !important;
        color: #000000 !important;
        border-color: #ffffff !important;
    }
    
    /* Expander 열렸을 때 내부 아이콘도 흰색 */
    details[open] > .streamlit-expanderHeader svg {
        stroke: #ffffff !important;
    }
    
    /* Expander 내용 부분 */
    .streamlit-expanderContent {
        background-color: #ffffff;
        border: 1px solid #e0e7ff;
        border-top: none;
        border-radius: 0 0 8px 8px;
    }
    
    /* ========== Markdown 텍스트 (흰 배경이므로 진한 글자) ========== */
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #334155;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #1e40af;
    }
    
    .stMarkdown strong {
        color: #1e293b;
    }
</style>
""",
    unsafe_allow_html=True,
)

df = pd.read_csv("./data/categories_select.csv")

# 🏥 메인 제목
st.markdown(
    '<h1 class="main-title">암 환자 고위험군 선별 및 예측 시스템</h1>',
    unsafe_allow_html=True,
)

# 📋 메인 탭 구성
tab1, tab2 = st.tabs(["환자 정보 입력", "샘플 예측"])

# ==================== 탭1: 환자 정보 입력 ====================
with tab1:
    st.markdown(
        '<div class="section-header"><h2>환자 정보 입력 및 예측</h2></div>',
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1, 2])

    with col_left:
        selected_values = {}

        # Primary Site - labeled 전용 처리
        if "Primary Site" in df.columns and "Primary Site - labeled" in df.columns:
            # 두 컬럼을 매핑 딕셔너리로 생성
            mapping = dict(zip(df["Primary Site - labeled"], df["Primary Site"]))

            # 라벨 목록을 unique하게 정렬
            unique_labels = sorted(
                df["Primary Site - labeled"].dropna().unique().tolist()
            )

            # 사용자에게 라벨을 selectbox로 보여주기
            selected_label = st.selectbox("🎯 Primary Site 선택", unique_labels)

            # 선택된 라벨에 해당하는 코드 자동 매칭
            selected_values["Primary Site - labeled"] = selected_label
            selected_values["Primary Site"] = mapping[selected_label]

        # 나머지 컬럼들 처리
        for col in df.columns:
            # Primary Site 관련 컬럼은 건너뛴다 (이미 처리했으므로)
            if col in ["Primary Site", "Primary Site - labeled"]:
                continue

            unique_vals = sorted(df[col].dropna().unique().tolist())

            if unique_vals:  # 값이 있으면 selectbox
                # 이모티콘 매핑
                emoji_map = {
                    "Age": "👤",
                    "Sex": "⚥",
                    "Race": "🌍",
                    "Stage": "📊",
                    "Grade": "📈",
                    "Tumor Size": "📏",
                    "Surgery": "🔪",
                    "Radiation": "☢️",
                    "Chemotherapy": "💊",
                }
                emoji = emoji_map.get(col, "📝")

                selected = st.selectbox(f"{emoji} {col} 선택", unique_vals)
                selected_values[col] = selected

        # 예측 버튼
        st.markdown("---")
        predict_button = st.button(
            "예측 실행", key="main_predict", use_container_width=True
        )

    with col_right:
        if not predict_button:
            st.markdown(
                f"""
            <div style="
                background-color: rgba(59, 130, 246, 0.1);
                padding: 12px 20px;
                border-radius: 8px; 
                color: #1e293b;
                font-size: 16px;
                margin-bottom: 16px;
            ">
                👈 왼쪽에서 환자 정보를 입력하고 '예측 실행' 버튼을 클릭하세요!
            </div>
            """,
                unsafe_allow_html=True,
            )

# ==================== 탭2: 샘플 예측 ====================
with tab2:
    st.markdown(
        '<div class="section-header"><h2>샘플 데이터 예측</h2></div>',
        unsafe_allow_html=True,
    )


sui_input_file_path = ["./data/Suicide.csv"]
sui_df = pd.read_csv(sui_input_file_path[0])
cols = sui_df.columns.tolist()
dtypes = sui_df.dtypes.to_dict()  # {col_name: dtype, ...}

# ==================== 탭1: 환자 정보 입력 예측 실행 ====================
if "predict_button" in locals() and predict_button:
    with tab1:
        with col_right:
            with st.spinner("AI가 예측을 수행 중입니다..."):
                # 기존 데이터셋 첫 행을 기반으로 input_df 생성
                input_df = sui_df.iloc[[0]].copy()  # 첫 행 복사, dtype 그대로 유지

                for col, val in selected_values.items():
                    if col in input_df.columns and val is not None:
                        input_df.at[0, col] = str(val)  # 무조건 str로 변환

                input_df_encoded = dp.run(input_df)

                # 예측 실행
                result_df = ModelAnalysis.predict_event_probabilities(
                    input_df=input_df, dp=dp, model=model, device=device
                )

                st.success("✅ 예측이 완료되었습니다!")

                # 상세 분석
                st.markdown("---")
                st.markdown("### 상세 생존 분석")
                ModelAnalysis.visualize_single_prediction(
                    input_df=input_df, dp=dp, model=model, device=device
                )

# ==================== 탭2: 샘플 예측 UI ====================
with tab2:
    col1, col2 = st.columns([1, 2])

    with col1:

        # 사건별 이모티콘 표시 매핑
        event_emoji_map = {
            -1: "생존",
            0: "암 관련 사망",
            1: "합병증 관련 사망",
            2: "기타 질환 사망",
            3: "자살/자해",
        }

        # 한글 이름을 보여주고 숫자로 역매핑
        event_display_options = {
            "생존": -1,
            "암 관련 사망": 0,
            "합병증 관련 사망": 1,
            "기타 질환 사망": 2,
            "자살/자해": 3,
        }

        # Streamlit selectbox로 event 라벨 선택 (한글 표시)
        selected_event_name = st.selectbox(
            "🎯예측할 사건 라벨 선택", list(event_display_options.keys())
        )

        # 선택된 한글 이름을 숫자로 변환
        selected_event_label = event_display_options[selected_event_name]

        st.markdown("---")
        sample_predict_button = st.button("샘플 예측 실행", use_container_width=True)

        st.markdown("---")
        with st.expander("📖 샘플 예측이란?"):
            st.markdown(
                """
            **실제 테스트 데이터셋**에서 특정 사건 라벨을 가진 샘플을 랜덤으로 선택하여 
            모델의 예측 성능을 확인할 수 있습니다.
            
            - **실제 관측 시간**: 환자가 실제로 사건을 경험한 시간
            - **실제 발생 사건**: 환자에게 실제로 발생한 사건
            
            예측 결과와 실제 값을 비교하여 모델의 정확도를 평가할 수 있습니다.
            """
            )

    with col2:
        if not sample_predict_button:

            st.markdown(
                f"""
            <div style="
                background-color: rgba(59, 130, 246, 0.1);
                padding: 12px 20px;
                border-radius: 8px; 
                color: #1e293b;
                font-size: 16px;
                margin-bottom: 16px;
            ">
                👈 왼쪽에서 사건 라벨을 선택하고 '샘플 예측 실행' 버튼을 클릭하세요! 
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            # st.markdown("### 샘플 예측 결과")
            with st.spinner("샘플 데이터 처리 중..."):
                # st.write(
                #     f"선택한 사건(event={selected_event_label}) 라벨에서 1개 샘플을 랜덤으로 선택하여 예측합니다..."
                # )

                # 🔹 test_dataset에서 선택한 event 샘플 인덱스 찾기
                indices = [
                    i
                    for i, (_, _, event) in enumerate(test_dataset)
                    if event == selected_event_label
                ]

                if not indices:
                    st.warning("선택한 사건 라벨에 해당하는 샘플이 없습니다.")
                else:
                    # 🔹 랜덤으로 하나 선택
                    import random

                    selected_idx = random.choice(indices)
                    x, time_val, event_val = test_dataset[selected_idx]

                    # 배치 차원 추가
                    sample_input = x.unsqueeze(0)  # shape: (1, num_features)

                    model.eval()
                    with torch.no_grad():
                        # 🔹 DataFrame 변환 (컬럼 이름 무시)
                        input_df_sample = pd.DataFrame(sample_input.numpy())

                        # 🔹 예측 실행
                        result_df = ModelAnalysis.predict_event_probabilities(
                            input_df=input_df_sample,
                            model=model,
                            device=device,
                            time_column="time",
                            target_column="event",
                        )

                    # st.success("✅ 샘플 예측 완료!")

                    # 실제 값 표시
                    col2_1, col2_2 = st.columns(2)
                    with col2_1:
                        st.metric(label="🕐 실제 관측 시간", value=f"{time_val} 개월")
                    with col2_2:
                        actual_event = event_emoji_map.get(
                            event_val, f"사건 {event_val}"
                        )
                        st.metric(label="📋 실제 발생 사건", value=actual_event)

                    # 상세 분석 차트
                    st.markdown("---")
                    st.markdown("### 📊 샘플 상세 분석")
                    ModelAnalysis.visualize_single_prediction(
                        input_df=input_df_sample,
                        model=model,
                        device=device,
                        time_column="time",
                        target_column="event",
                        event_weights=[3.0, 5.0, 5.0, 10.0],
                    )
