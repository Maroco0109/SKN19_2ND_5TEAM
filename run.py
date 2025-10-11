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

import torch

import modules.DataAnalysis as DataAnalysis
import modules.ModelAnalysis as ModelAnalysis
import modules.DataModify as DataModify
from modules.DataSelect import DataPreprocessing

import modules.Models as Models

input_dim = 17                      # input dimension : data의 feature의 개수
hidden_size = (128, 64)             # 1번째, 2번째 hidden layer의 size
time_bins = 91                      # 3개월 단위로 time을 split하여 각 구간으로 삼음 -> 최대 270개월 + 그 후
num_events = 4                      # 사건의 개수

input_params_path = "./parameters/deephit_model_feature.pth"
device = torch.device("cpu")

categories = DataPreprocessing.load_category()
dp = DataPreprocessing(categories=categories)

# 모델 정의 (학습할 때 사용한 모델 클래스)
model = Models.DeepHitSurvWithSEBlock(input_dim, 
                    hidden_size, 
                    time_bins, 
                    num_events,
                    )  # 사건 수 맞게 설정
model.load_state_dict(torch.load(input_params_path, map_location=device))
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

# 예측 버튼
if st.button("예측 실행"):
    
    input_df = pd.DataFrame([{col: 0 for col in cols}])

    # selected_values = {'Age': 65, 'Gender': 'Male', ...}
    for col, val in selected_values.items():
        if col in input_df.columns:
            input_df.at[0, col] = val  # 0행(col 위치)에 값 덮어쓰기

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