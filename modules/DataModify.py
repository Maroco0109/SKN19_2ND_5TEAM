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

        # SEER Summary Stage 확장 지역 코드(2004+) 컬럼명 상수
        self.SEER_SUMMARY_STAGE_COL = 'Combined Summary Stage with Expanded Regional Codes (2004+)'

    @staticmethod
    def drop_cols(self, df, cols=None) :
        if df is None:
            raise ValueError('DataFrame cannot be None when dropping columns')
        if not cols:
            return df.copy()
        return df.drop(columns=list(cols), errors='ignore')

    # categorical한 데이터 encoding (일관된 매핑 유지)
    def category_encoding(
        self,
        df: pd.DataFrame,
        categories: Optional[Dict[str, Dict[str, int]]] = None,
        encoding: str = 'label'
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
        if df is None:
            raise ValueError('Input DataFrame is required for category encoding')

        categories = {**(categories or {})}
        # SEER Summary Stage 컬럼 표준화(요청된 매핑 적용)
        df_standardized = self._normalize_seer_summary_stage(df)
        categorical_col = DataSelect.return_cols(df_standardized, 'categorical', boundary=100)
        df_encoded = df_standardized.copy()

        if encoding == 'label':
            categories['encoding_type'] = 'label'
            for col in categorical_col:
                # 숫자형(이미 인코딩된) 컬럼은 건너뜀
                if pd.api.types.is_numeric_dtype(df_encoded[col]):
                    continue
                series = df_encoded[col].astype('object')
                # 기존 매핑을 유지하고, 새로운 값만 뒤에 추가합니다.
                existing = categories.get(col, {})
                label_map = {k: v for k, v in existing.items() if isinstance(v, int) and v >= 0}
                next_id = (max(label_map.values()) + 1) if label_map else 0
                for val in series.dropna().unique():
                    if val not in label_map:
                        label_map[val] = next_id
                        next_id += 1
                df_encoded[col] = series.map(label_map).fillna(-1).astype(int)
                label_map['__MISSING__'] = -1
                categories[col] = label_map

        elif encoding == 'onehot':
            categories['encoding_type'] = 'onehot'
            for col in categorical_col:
                # 숫자형(이미 인코딩된) 컬럼은 건너뜀
                if pd.api.types.is_numeric_dtype(df_encoded[col]):
                    continue
                series = df_encoded[col].astype('object')
                # 기존 카테고리 목록 유지 + 신규 값 추가
                existing_cols = categories.get(col)
                if isinstance(existing_cols, list):
                    # 기존 리스트는 dummy 컬럼명(prefix 포함)일 수 있으므로 카테고리 이름만 복원
                    prefix = f"{col}_"
                    known = [c[len(prefix):] if c.startswith(prefix) else c for c in existing_cols]
                else:
                    known = []
                for val in series.dropna().unique():
                    if val not in known:
                        known.append(val)
                # 카테고리 dtype으로 고정 후 더미 생성(예상 컬럼 모두 포함하도록 후처리)
                cat = pd.Categorical(series, categories=known)
                dummies = pd.get_dummies(cat, prefix=col)
                expected_cols = [f"{col}_{v}" for v in known]
                for expect in expected_cols:
                    if expect not in dummies.columns:
                        dummies[expect] = 0
                dummies = dummies[expected_cols]
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
                categories[col] = expected_cols

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

    # SEER Summary Stage(2004+)를 0/1/2/3/9로 표준화(정수 인코딩)
    def _normalize_seer_summary_stage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        'Combined Summary Stage with Expanded Regional Codes (2004+)' 컬럼을
        다음 규칙으로 통일합니다.
          - In situ            -> 0
          - Localized          -> 1
          - Regional (모든 세부 분류 포함) -> 2
          - Distant            -> 3
          - Unknown/Unstaged/N/A/Blank 등 -> 9

        원본 컬럼이 없으면 그대로 반환합니다.
        """
        col = self.SEER_SUMMARY_STAGE_COL
        if col not in df.columns:
            return df

        def map_stage(val) -> int:
            if pd.isna(val):
                return 9
            s = str(val).strip().lower()
            if s == 'in situ' or 'in situ' in s:
                return 0
            if s.startswith('localized') or s == 'localized':
                return 1
            if s.startswith('regional'):
                # regional by direct extension only, by lymph nodes only, both, nos 등 포함
                return 2
            if s.startswith('distant') or s == 'distant':
                return 3
            # Unknown/unstaged/blank/not applicable 등은 9
            unknown_keys = ['unknown', 'unstaged', 'not applicable', 'blank', 'n/a', 'na', 'not known']
            if any(k in s for k in unknown_keys) or s in {'', '.'}:
                return 9
            # 기타 예외값은 보수적으로 9 처리
            return 9

        df_out = df.copy()
        df_out[col] = df_out[col].map(map_stage)
        # 명시적으로 정수형 보장
        df_out[col] = pd.to_numeric(df_out[col], errors='coerce').fillna(9).astype(int)
        return df_out

    # 순서형 컬럼을 지정된 규칙에 따라 정수 라벨로 변환
    def encode_ordinal_columns(
        self,
        df: pd.DataFrame,
        ordinal_config: Optional[Dict[str, Iterable]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
        """
        순서형 컬럼을 일관된 규칙으로 정수 라벨로 변환합니다.
        - definition == 'numeric' 인 경우: 순위 인덱스가 아닌 원래의 수치값을 그대로 사용해 일관성을 보장합니다.
        - definition 이 리스트인 경우: 정의된 전체 순서를 매핑으로 사용하여, 데이터셋마다 등장 여부와 무관하게 같은 값은 같은 정수로 인코딩됩니다.
        """
        config = ordinal_config or self.ORDINAL_CONFIG_DEFAULT
        df_encoded = df.copy()
        mappings: Dict[str, Dict[str, int]] = {}

        for col, definition in config.items():
            if col not in df_encoded.columns:
                continue
            series = df_encoded[col]

            if isinstance(definition, str):
                if definition != 'numeric':
                    raise ValueError(f'Unsupported ordinal definition: {definition}')
                # 수치형으로 직접 변환하여 사용 (순위가 아닌 절대값 보존)
                df_encoded[col] = pd.to_numeric(series, errors='coerce')
                mappings[col] = {'__TYPE__': 'numeric'}
            else:
                # 사전 정의된 전체 순서를 그대로 매핑으로 사용
                order = list(definition)
                mapping = {value: idx for idx, value in enumerate(order)}
                mapping['__MISSING__'] = -1
                df_encoded[col] = series.map(mapping).fillna(-1).astype(int)
                mappings[col] = mapping

        return df_encoded, mappings

    # 명목형 컬럼을 팩터라이즈하여 정수형으로 변환
    @staticmethod
    def encode_nominal_columns(
        df: pd.DataFrame,
        exclude_columns: Optional[Iterable[str]] = None,
        existing_mappings: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
        """
        명목형 컬럼을 정수로 인코딩. 기존 매핑이 주어지면 그대로 유지하면서 신규 값만 뒤에 추가합니다.
        결측/미등록 값은 -1로 치환합니다.
        """
        excludes = set(exclude_columns or [])
        df_encoded = df.copy()
        mappings: Dict[str, Dict[str, int]] = {}
        existing_mappings = existing_mappings or {}

        for col in df_encoded.columns:
            if col in excludes or pd.api.types.is_numeric_dtype(df_encoded[col]):
                continue
            series = df_encoded[col].astype('object')

            prev = existing_mappings.get(col, {})
            value_map = {k: v for k, v in prev.items() if isinstance(v, int) and v >= 0}
            next_id = (max(value_map.values()) + 1) if value_map else 0

            for val in series.dropna().unique():
                if val not in value_map:
                    value_map[val] = next_id
                    next_id += 1

            df_encoded[col] = series.map(value_map).fillna(-1).astype(int)
            value_map['__MISSING__'] = -1
            mappings[col] = value_map

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
        drop_label_source: bool = True,
        existing_meta: Optional[Dict[str, Dict]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Dict]]:
        # 먼저 SEER Summary Stage 컬럼을 요구된 규칙으로 표준화
        df_work = self._normalize_seer_summary_stage(df)

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

        prev_nominal = None
        if existing_meta and isinstance(existing_meta.get('nominal_mappings'), dict):
            prev_nominal = existing_meta.get('nominal_mappings')
        df_work, nominal_mappings = self.encode_nominal_columns(
            df_work,
            exclude_columns=exclude,
            existing_mappings=prev_nominal,
        )
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
        drop_label_source: bool = True,
        existing_meta: Optional[Dict[str, Dict]] = None,
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
            existing_meta=existing_meta or self.meta,
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
            transformed = self.transform(df)
            if isinstance(transformed, tuple):
                df = transformed[0]
            else:
                df = transformed

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

def decode_csv_features(csv_path, categories):
    """
    CSV에 저장된 인코딩된 feature 데이터를 categories 딕셔너리로 디코딩

    Args:
        csv_path (str): 인코딩된 CSV 파일 경로
        categories (dict): DataPreprocessing.categories 딕셔너리

    Returns:
        pd.DataFrame: 디코딩된 feature 데이터
    """
    # CSV 불러오기
    df_encoded = pd.read_csv(csv_path)

    # 디코딩할 컬럼 이름 순서
    feature_cols = [col for col in df_encoded.columns if col != 'encoding_type']

    df_decoded = pd.DataFrame(columns=feature_cols)

    for col in feature_cols:
        if col in categories:
            inverse_map = {v: k for k, v in categories[col].items()}
            df_decoded[col] = df_encoded[col].apply(lambda x: inverse_map.get(int(x), x))
        else:
            # 숫자형 컬럼이면 그대로 사용
            df_decoded[col] = df_encoded[col]

    return df_decoded
