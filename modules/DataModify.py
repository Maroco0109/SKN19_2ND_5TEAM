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

        if self.transform is not None:
            df = self.transform(df)

        self.time = df[time_column].values.astype(int) if time_column else None
        self.target = df[target_column].values.astype(int) if target_column else None

        drop_cols = [col for col in [time_column, target_column] if col is not None]
        self.data = df.drop(columns=drop_cols).values.astype(float)

        self.data = torch.tensor(self.data, dtype=torch.float32)
        if self.target is not None:
            self.target = torch.tensor(self.target, dtype=torch.int64)
        if self.time is not None:
            self.time = torch.tensor(self.time, dtype=torch.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        t = self.time[index] if self.time is not None else None
        y = self.target[index] if self.target is not None else None
        return x, t, y
    

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
