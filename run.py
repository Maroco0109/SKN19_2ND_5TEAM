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

import modules.DataAnalysis as DataAnalysis
import modules.ModelAnalysis as ModelAnalysis
import modules.DataModify as DataModify
from modules.DataSelect import DataPreprocessing

import modules.Models as Models

# Dataset 로드
test_file = ['./data/test dataset_fixed.csv']
test_dataset = DataModify.CancerDataset(
    target_column='event',
    time_column='time',
    file_paths=test_file
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

input_dim = 17                      # input dimension : data의 feature의 개수
hidden_size = (128, 64)             # 1번째, 2번째 hidden layer의 size
time_bins = 91                      # 3개월 단위로 time을 split하여 각 구간으로 삼음 -> 최대 270개월 + 그 후
num_events = 4                      # 사건의 개수

input_params_path = "./parameters/deephit_model_2D_CNN.pth"
device = torch.device("cpu")

encoding_map = DataPreprocessing.load_category()
print(type(encoding_map))

# 예시: 모든 값도 문자열로 변환하려면 convert_values_to_str=True
str_encoding_map = ModelAnalysis.clean_encoding_map(encoding_map, convert_values_to_str=True)

dp = DataPreprocessing(categories=str_encoding_map)

# 모델 정의 (학습할 때 사용한 모델 클래스)
model = Models.DeepHitSurvWithSEBlockAnd2DCNN(input_dim, 
                    hidden_size, 
                    time_bins, 
                    num_events,
                    )  # 사건 수 맞게 설정
model.load_state_dict(torch.load(input_params_path, map_location=device, weights_only=True))
model.to(device)
model.eval()  # 평가 모드

df = pd.read_csv('./data/categories_select.csv')

st.title("암 환자 고위험군 선별 및 예측 시스템")

selected_values = {}

# Primary Site - labeled 전용 처리
if "Primary Site" in df.columns and "Primary Site - labeled" in df.columns:
    # 두 컬럼을 매핑 딕셔너리로 생성
    mapping = dict(zip(df["Primary Site - labeled"], df["Primary Site"]))

    # 라벨 목록을 unique하게 정렬
    unique_labels = sorted(df["Primary Site - labeled"].dropna().unique().tolist())

    # 사용자에게 라벨을 selectbox로 보여주기
    selected_label = st.selectbox("Primary Site 선택", unique_labels)

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
        selected = st.selectbox(f"{col} 선택", unique_vals)
        selected_values[col] = selected

sui_input_file_path = ['./data/Suicide.csv']
sui_df = pd.read_csv(sui_input_file_path[0])
cols = sui_df.columns.tolist()
dtypes = sui_df.dtypes.to_dict()  # {col_name: dtype, ...}

# 예측 버튼
if st.button("예측 실행"):
    
    # 기존 데이터셋 첫 행을 기반으로 input_df 생성
    input_df = sui_df.iloc[[0]].copy()  # 첫 행 복사, dtype 그대로 유지

    for col, val in selected_values.items():
        if col in input_df.columns and val is not None:
            input_df.at[0, col] = str(val)  # 무조건 str로 변환

    input_df_encoded = dp.run(input_df)

    print(input_df_encoded)

    # 예측 실행
    result_df = ModelAnalysis.predict_event_probabilities(
        input_df=input_df,
        dp=dp,
        model=model,
        device=device
    )

    ModelAnalysis.visualize_single_prediction(
        input_df=input_df,
        dp=dp,
        model=model,
        device=device
    )

# Streamlit selectbox로 event 라벨 선택
all_events = sorted(list(set([label for _, _, label in test_dataset])))
selected_event_label = st.selectbox(
    "예측할 사건(event) 라벨 선택",
    [-1, 0, 1, 2, 3]
)

if st.button("샘플 예측 실행"):
    st.write(f"선택한 사건(event={selected_event_label}) 라벨에서 1개 샘플을 랜덤으로 선택하여 예측합니다...")

    # 🔹 test_dataset에서 선택한 event 샘플 인덱스 찾기
    indices = [i for i, (_, _, event) in enumerate(test_dataset) if event == selected_event_label]

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
            input_df = pd.DataFrame(sample_input.numpy())

            # 🔹 예측 실행
            result_df = ModelAnalysis.predict_event_probabilities(
                input_df=input_df,
                model=model,
                device=device,
                time_column='time',
                target_column='event'
            )

            # 🔹 시각화
            ModelAnalysis.visualize_single_prediction(
                input_df=input_df,
                model=model,
                device=device,
                time_column='time',
                target_column='event'
            )

            # 🔹 실제 값 출력
            st.markdown("---")
            st.subheader("📘 실제 값 정보")
            st.write(f"**실제 관측 시간 (time):** {time_val}")
            st.write(f"**실제 사건 (event):** {event_val}")
