"""

모델 분석 시각화 모듈

- fit된 모델을 인자로 받아서 수행

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch

def predict_event_probabilities(
    input_df: pd.DataFrame,
    dp,
    model,
    device: torch.device,
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
    # -----------------------------
    # 1️⃣ 전처리
    processed_df = dp.run(input_df)

    # -----------------------------
    # 2️⃣ feature만 추출
    drop_cols = [col for col in [time_column, target_column] if col in processed_df.columns]
    features_df = processed_df.drop(columns=drop_cols)

    # -----------------------------
    # 3️⃣ torch.tensor 변환
    x = torch.tensor(features_df.values.astype(float), dtype=torch.float32).to(device)

    # -----------------------------
    # 4️⃣ 모델 예측
    model.eval()
    with torch.no_grad():
        _, pmf, cif = model(x)  # cif: (1, num_events, time_bins)

    # -----------------------------
    # 5️⃣ 마지막 시간 bin 제거 (dummy)
    cif = cif[:, :, :-1]  # shape: (1, num_events, time_bins-1)

    # -----------------------------
    # 6️⃣ 사건별 × 시간별 DataFrame 생성
    num_events, num_time = cif.shape[1], cif.shape[2]
    cif_array = cif[0].cpu().numpy()  # (num_events, time_bins-1)

    time_points = [f"Time_{t}" for t in range(num_time)]
    columns = pd.MultiIndex.from_product(
        [[f"Event_{i}" for i in range(num_events)], time_points],
        names=['Event', 'Time']
    )

    result_df = pd.DataFrame(cif_array.flatten()[None, :], columns=columns)

    return result_df

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


# 예시
def show_model_graph(model, x, y, e, cols) :
    pass