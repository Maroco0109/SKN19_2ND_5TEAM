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


# 예시
def show_model_graph(model, x, y, e, cols) :
    pass