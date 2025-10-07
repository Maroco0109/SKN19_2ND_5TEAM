"""

데이터 시각화 모듈

- 모든 데이터 시각화 코드를 함수로 구현
- 모든 함수는 분석할 데이터를 df 형태로 받음
- 기본적으로는 return 값이 없는 함수. 필요에 따라 몇몇 데이터를 df 형태로 return

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display 

import modules.DataModify as DataModify

# 데이터 프레임에서, 범주형 데이터에 속하는 값들의 컬럼값을 출력
def show_value_counts(df, cols=None, boundary=30) :
    for col in df.columns:
        if df[col].nunique(dropna=True) > boundary :  # 각기 다른 값이 boundary 이상인 Continuous 한 값들은 출력하지 않음
            print(col)
            print('continuous')
            print("-"*20)
            continue

        series = df[col]
        if pd.api.types.is_extension_array_dtype(series.dtype):  # Nullable dtypes (e.g., Int64) need object conversion
            series = series.astype('object')
        value_counts = series.fillna("NA").value_counts(dropna=False)  # 결측치는 NA로 처리 후 출력
        print(value_counts)
        print("-" * 20)

# 예시 코드
def show_graph(df) :
    pass

