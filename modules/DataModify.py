"""

데이터에 대한 처리를 수행하는 모듈

1. 데이터 스플릿 하는 함수
2. 데이터 전처리 클래스
    - 메소드로 각각의 과정 구현 후 run() 메소드로 일괄 적용
3. 전처리된 데이터 저장 코드

"""
import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader

class DataPreprocessing() :
    def __init__(self, df=None) :
        self.raw_data = df  # 수정 전의 원본 데이터를 저장해둠
    
    def drop_cols(self, df, cols=None) :
        pass

    """
    모든 전처리 기능 함수로 구현
    """

    def run(self, df, encoding='label') :
        df_cleaned = self.drop_cols(df)
        # df_cleaned = self.func(df_cleaned)
        # ...
        return df_cleaned   # 전처리가 완료된 df 반환

# 모델에 사용할 Dataset 형태   
class CancerDataset(Dataset) :
    def __init__(self, target_column=None, time_column=None, file_paths=None, transform=None) :
        df = load_data(file_paths)

        self.transform = transform

        df = self.transform(df)

        if time_column is None :
            self.data = df.values.astype(float)
            self.time = None
        else:
            self.time = df[time_column].values.astype(float)
            self.data = df.drop(columns=[time_column]).values.astype(float)

        if target_column is None:
            self.data = df.values.astype(float)
            self.target = None
        else:
            self.target = df[target_column].values.astype(float)
            self.data = df.drop(columns=[target_column]).values.astype(float)

        self.data = torch.tensor(self.data, dtype=torch.float32)
        if self.target is not None:
            self.target = torch.tensor(self.target, dtype=torch.float32)

    def __len__(self) :
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.time[index], self.target[index]
    

def load_data(file_paths) :
    if file_paths is None :
        input_file_path1 = './data/2022Data_part1.csv'
        input_file_path2 = './data/2022Data_part2.csv'
        file_paths = [input_file_path1, input_file_path2]

    df_list = []
    for path in file_paths:
        df = pd.read_csv(path)
        df_list.append(df)
    
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def split_data_X_y(df) :        # 데이터를 특성, 라벨(시간, 사건)으로 분리
    pass

def split_data_X_y_e(df) :      # 데이터를 특성, 시간, 사건으로 분리
    pass
