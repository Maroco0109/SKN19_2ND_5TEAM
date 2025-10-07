"""

모델 분석 시각화 모듈

- fit된 모델을 인자로 받아서 수행

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch


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

    # ----- 1️⃣ 생존 확률 -----
    survival = 1 - torch.sum(cif, dim=1)  # (B, T)

    # ----- 2️⃣ 위험도 (전체 사건 발생 확률 합) -----
    risk_score = pmf.sum(dim=(1, 2))  # (B,)

    # ----- 3️⃣ 기대 생존 시간 -----
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