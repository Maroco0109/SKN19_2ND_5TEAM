"""

데이터에 대한 처리를 수행하는 모듈

1. 데이터 스플릿 하는 함수
2. 데이터 전처리 클래스
    - 메소드로 각각의 과정 구현 후 run() 메소드로 일괄 적용
3. 전처리된 데이터 저장 코드

"""
import torch
import pandas as pd

from typing import Dict, Iterable, List, Optional, Tuple

from torch.utils.data import Dataset, DataLoader

from modules import DataSelect

class DataPreprocessing() :
    # 순서형 인코딩에 활용할 기본 나이 구간 정의
    AGE_RECODE_ORDER: List[str] = [
        '00 years', '01-04 years', '05-09 years', '10-14 years', '15-19 years',
        '20-24 years', '25-29 years', '30-34 years', '35-39 years', '40-44 years',
        '45-49 years', '50-54 years', '55-59 years', '60-64 years', '65-69 years',
        '70-74 years', '75-79 years', '80-84 years', '85-89 years', '90+ years'
    ]

    # 기본 순서형 컬럼 설정: 지정된 순서 혹은 숫자형으로 처리
    ORDINAL_CONFIG_DEFAULT: Dict[str, Iterable] = {
        'Age recode with <1 year olds and 90+': AGE_RECODE_ORDER,
        'Year of diagnosis': 'numeric',
        'Year of follow-up recode': 'numeric',
    }

    # 사망 원인 그룹을 최종 타깃 라벨로 매핑
    COD_GROUP_TO_TARGET: Dict[int, int] = {0: 0, 1: 1, 2: 2, 3: -1, 4: 3}

    # 타깃 라벨에 대한 설명 정보
    TARGET_DESCRIPTION: Dict[int, str] = {
        -1: 'Alive or external cause',
        0: 'Cancer-related death',
        1: 'Complication-related death',
        2: 'Other disease-related death',
        3: 'Suicide or self-inflicted',
    }

    def __init__(self, df=None) :
        self.raw_data = df  # 수정 전의 원본 데이터를 저장해둠
        self.categories: Dict[str, Dict[str, int]] = {}
        self.survival_flag_group_map = self._build_group_map(DataSelect.label_Surv_flags)
        self.cod_group_map = self._build_group_map(DataSelect.label_cod_list)
        self.meta: Optional[Dict[str, Dict]] = None
        self.encoded_df: Optional[pd.DataFrame] = None
        self.target: Optional[pd.Series] = None

    def drop_cols(self, df, cols=None) :
        if df is None:
            raise ValueError('DataFrame cannot be None when dropping columns')
        if not cols:
            return df.copy()
        return df.drop(columns=list(cols), errors='ignore')

    """
    모든 전처리 기능 함수로 구현
    """

    # categorical한 데이터 encoding
    def category_encoding(self, df: pd.DataFrame, categories: Optional[Dict[str, Dict[str, int]]] = None, encoding: str = 'label') -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
        if df is None:
            raise ValueError('Input DataFrame is required for category encoding')

        categories = {**categories} if categories else {}
        categorical_col = DataSelect.return_cols(df, 'categorical', boundary=100)
        df_encoded = df.copy()

        if encoding == 'label':
            categories['encoding_type'] = 'label'
            for col in categorical_col:
                unique_vals = df_encoded[col].dropna().unique()
                label_map = {val: idx for idx, val in enumerate(unique_vals)}
                df_encoded[col] = df_encoded[col].map(label_map)
                categories[col] = label_map

        elif encoding == 'onehot':
            categories['encoding_type'] = 'onehot'
            for col in categorical_col:
                dummies = pd.get_dummies(df_encoded[col], prefix=col)
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
                categories[col] = dummies.columns.tolist()

        else:
            raise ValueError(f'알 수 없는 encoding_type: {encoding}')

        return df_encoded, categories

    # encoding된 데이터 decoding
    def category_decoding(self, df: pd.DataFrame, categories: Dict[str, Dict[str, int]]) -> pd.DataFrame:
        if df is None:
            raise ValueError('Input DataFrame is required for category decoding')
        if not categories:
            raise ValueError('Categories mapping is required for decoding')

        df_decoded = df.copy()
        encoding_type = categories.get('encoding_type')

        if encoding_type == 'label':
            for col, mapping in categories.items():
                if col == 'encoding_type':
                    continue
                reverse_map = {v: k for k, v in mapping.items()}
                df_decoded[col] = df_decoded[col].map(reverse_map)

        elif encoding_type == 'onehot':
            for col, dummy_cols in categories.items():
                if col == 'encoding_type':
                    continue

                existing_cols = [dummy for dummy in dummy_cols if dummy in df_decoded.columns]

                def decode_row(row):
                    for dummy_col in existing_cols:
                        if row.get(dummy_col, 0) == 1:
                            return dummy_col.replace(f'{col}_', '')
                    return None

                df_decoded[col] = df_decoded.apply(decode_row, axis=1)
                if existing_cols:
                    df_decoded = df_decoded.drop(columns=existing_cols)

        else:
            raise ValueError(f'알 수 없는 encoding_type: {encoding_type}')

        return df_decoded

    # 생존 플래그/사인 그룹 정의를 인덱스로 치환
    @staticmethod
    def _build_group_map(definitions: Iterable[Iterable[str]]) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        for idx, group in enumerate(definitions):
            for value in group:
                mapping[value] = idx
        return mapping

    # 생존 개월 수를 구간화해 파생 변수 생성
    @staticmethod
    def bin_survival_months(series: pd.Series, bin_size: int = 3) -> pd.Series:
        if bin_size <= 0:
            raise ValueError('bin_size must be a positive integer')
        numeric = pd.to_numeric(series, errors='coerce')
        # 모델 안정성을 위해 270개월(=90개 3개월 구간) 이상은 270으로 상한을 둡니다.
        numeric = numeric.clip(upper=270)
        binned = (numeric // bin_size).astype('Int64')
        binned = binned.where(~numeric.isna(), other=pd.NA)
        return binned.fillna(-1).astype(int)

    # 순서형 컬럼을 지정된 규칙에 따라 정수 라벨로 변환
    def encode_ordinal_columns(self, df: pd.DataFrame, ordinal_config: Optional[Dict[str, Iterable]] = None) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
        config = ordinal_config or self.ORDINAL_CONFIG_DEFAULT
        df_encoded = df.copy()
        mappings: Dict[str, Dict[str, int]] = {}

        for col, definition in config.items():
            if col not in df_encoded.columns:
                continue
            series = df_encoded[col]
            present_values = series.dropna().unique()

            if isinstance(definition, str):
                if definition != 'numeric':
                    raise ValueError(f'Unsupported ordinal definition: {definition}')
                numeric_series = pd.to_numeric(pd.Series(present_values), errors='coerce')
                if numeric_series.isna().all():
                    order = sorted(present_values, key=lambda x: str(x))
                else:
                    ordered_pairs = sorted(
                        [(num, val) for num, val in zip(numeric_series, present_values) if not pd.isna(num)],
                        key=lambda pair: pair[0]
                    )
                    order = [val for _, val in ordered_pairs]
                    non_numeric_vals = [val for num, val in zip(numeric_series, present_values) if pd.isna(num)]
                    order.extend(sorted(non_numeric_vals, key=lambda x: str(x)))
            else:
                present_set = set(present_values)
                order = [value for value in definition if value in present_set]
                remaining = sorted(present_set - set(order), key=str)
                order.extend(remaining)

            mapping = {value: idx for idx, value in enumerate(order)}
            mapping['__MISSING__'] = -1
            df_encoded[col] = series.map(mapping).fillna(-1).astype(int)
            mappings[col] = mapping

        return df_encoded, mappings

    # 명목형 컬럼을 팩터라이즈하여 정수형으로 변환
    @staticmethod
    def encode_nominal_columns(df: pd.DataFrame, exclude_columns: Optional[Iterable[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
        excludes = set(exclude_columns or [])
        df_encoded = df.copy()
        mappings: Dict[str, Dict[str, int]] = {}

        for col in df_encoded.columns:
            if col in excludes or pd.api.types.is_numeric_dtype(df_encoded[col]):
                continue
            series = df_encoded[col].astype('object')
            codes, uniques = pd.factorize(series, sort=True)
            df_encoded[col] = codes
            mapping = {value: idx for idx, value in enumerate(uniques)}
            mapping['__MISSING__'] = -1
            mappings[col] = mapping

        return df_encoded, mappings

    # 생존/사망 정보를 결합해 다중 클래스 타깃을 생성
    def create_combined_label(self, df: pd.DataFrame, cod_col: str = 'COD to site recode', survival_flag_col: str = 'Survival months flag', vital_status_col: str = 'Vital status recode (study cutoff used)') -> Tuple[pd.Series, pd.Series, Dict[int, str]]:
        if survival_flag_col not in df or vital_status_col not in df or cod_col not in df:
            missing = [col for col in [cod_col, survival_flag_col, vital_status_col] if col not in df]
            raise KeyError(f'Missing required columns: {missing}')

        survival_groups = df[survival_flag_col].map(self.survival_flag_group_map)
        vital_status = df[vital_status_col]

        drop_mask = (
            survival_groups.isna()
            | (survival_groups.eq(2) & vital_status.eq('Dead'))
            | survival_groups.eq(3)
        )
        valid_mask = ~drop_mask

        labels = pd.Series(pd.NA, index=df.index, dtype='Int64')
        alive_mask = valid_mask & vital_status.eq('Alive')
        labels.loc[alive_mask] = -1

        dead_mask = valid_mask & vital_status.eq('Dead')
        if dead_mask.any():
            cod_groups = df.loc[dead_mask, cod_col].map(self.cod_group_map)
            labels.loc[dead_mask] = cod_groups.map(self.COD_GROUP_TO_TARGET)

        final_mask = labels.notna()
        labels = labels.loc[final_mask].astype(int)
        labels.name = 'target_label'
        return labels, final_mask, self.TARGET_DESCRIPTION

    # 전체 전처리 파이프라인을 실행해 학습 데이터를 구성
    def preprocess_for_model(
        self,
        df: pd.DataFrame,
        ordinal_config: Optional[Dict[str, Iterable]] = None,
        bin_size: int = 3,
        survival_months_col: str = 'Survival months',
        cod_col: str = 'COD to site recode',
        survival_flag_col: str = 'Survival months flag',
        vital_status_col: str = 'Vital status recode (study cutoff used)',
        drop_label_source: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Dict]]:
        df_work = df.copy()

        labels, valid_mask, label_desc = self.create_combined_label(
            df_work,
            cod_col=cod_col,
            survival_flag_col=survival_flag_col,
            vital_status_col=vital_status_col,
        )
        df_work = df_work.loc[labels.index].copy()
        df_work['target_label'] = labels.astype(int)

        bin_col_name = None
        if survival_months_col in df_work.columns:
            df_work[survival_months_col] = pd.to_numeric(df_work[survival_months_col], errors='coerce')
            # 구간화와 일관성을 위해 원본 생존 개월도 270개월로 상한을 둡니다.
            df_work[survival_months_col] = df_work[survival_months_col].clip(upper=270)
            bin_col_name = f'{survival_months_col}_bin_{bin_size}m'
            df_work[bin_col_name] = self.bin_survival_months(df_work[survival_months_col], bin_size=bin_size)

        df_work, ordinal_mappings = self.encode_ordinal_columns(df_work, ordinal_config)

        exclude = set((ordinal_config or self.ORDINAL_CONFIG_DEFAULT).keys())
        exclude.add('target_label')
        if survival_months_col in df_work.columns:
            exclude.add(survival_months_col)
        if bin_col_name:
            exclude.add(bin_col_name)
        if drop_label_source:
            for col in [cod_col, survival_flag_col, vital_status_col]:
                if col in df_work.columns:
                    df_work = df_work.drop(columns=col)
        else:
            exclude.update([cod_col, survival_flag_col, vital_status_col])

        df_work, nominal_mappings = self.encode_nominal_columns(df_work, exclude_columns=exclude)
        df_work = df_work.reset_index(drop=True)

        meta: Dict[str, Dict] = {
            'ordinal_mappings': ordinal_mappings,
            'nominal_mappings': nominal_mappings,
            'label_description': label_desc,
            'label_column': 'target_label',
            'survival_bin_column': bin_col_name,
            'bin_size_months': bin_size,
            'retained_mask': valid_mask,
            'retained_index': labels.index,
        }

        return df_work, df_work['target_label'], meta

    def run(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        mode: str = 'full',
        encoding: str = 'label',
        categories: Optional[Dict[str, Dict[str, int]]] = None,
        ordinal_config: Optional[Dict[str, Iterable]] = None,
        bin_size: int = 3,
        survival_months_col: str = 'Survival months',
        cod_col: str = 'COD to site recode',
        survival_flag_col: str = 'Survival months flag',
        vital_status_col: str = 'Vital status recode (study cutoff used)',
        drop_label_source: bool = True
    ):
        df_input = df if df is not None else self.raw_data
        if df_input is None:
            raise ValueError('No DataFrame provided to run preprocessing')

        if mode == 'simple':
            encoded_df, categories_map = self.category_encoding(df_input, categories, encoding)
            self.encoded_df = encoded_df
            self.categories = categories_map
            return encoded_df, categories_map

        if mode == 'decode':
            decode_map = categories or self.categories
            if not decode_map:
                raise ValueError('Decoding requires a categories mapping')
            return self.category_decoding(df_input, decode_map)

        processed_df, target, meta = self.preprocess_for_model(
            df_input,
            ordinal_config=ordinal_config,
            bin_size=bin_size,
            survival_months_col=survival_months_col,
            cod_col=cod_col,
            survival_flag_col=survival_flag_col,
            vital_status_col=vital_status_col,
            drop_label_source=drop_label_source,
        )
        self.encoded_df = processed_df
        self.meta = meta
        self.target = target
        return processed_df, target, meta

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
            self.target = torch.tensor(self.target, dtype=torch.long)
        if self.time is not None:
            self.time = torch.tensor(self.time, dtype=torch.long)

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
