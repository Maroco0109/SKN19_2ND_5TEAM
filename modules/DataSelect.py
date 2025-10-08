import numpy as np
import pandas as pd

"""

특정 조건을 만족하는 데이터를 select 하기 위한 조건 설정

** COD to site recode, Survival months flag, Vital status recode 를 결합하여 하나의 라벨로 완성 **

1. label_Surv_flags 를 이용해 일부 데이터 드랍
    1., 2.에 속하지만 Alive인 데이터는 그대로 사용
    2. 에 속하지만 dead인 데이터, 3. 에 속하는 데이터는 드랍

2. 남은 데이터들을 이용해 사건(사인) 별로 라벨링
    생존 : -1
    label_cod_list에서 4번 라벨(사고사)에 해당 : -1
    암 관련으로 사망 : 0
    합병증 관련으로 사망 : 1
    기타 질병으로 사망 : 2
    자해 및 자살로 인한 사망 : 3


"""

class DataPreprocessing2:
    def __init__(self, drop_cols=None, label_cols=None, boundary=100):
        # 인코딩 사전
        self.categories = {}            # {col: {category_value: integer_label(>=1)}}
        self.categories_meta = {        # 메타
            'encoding_type': 'label',
            'nan_label': 0,
            'starts_from': 1,
            'boundary': boundary
        }
        # 드랍/라벨 컬럼
        self.drop_cols = drop_cols or []
        self.label_cols = label_cols or []   # 라벨 생성에 사용할 컬럼(인코딩 제외)
        self.fitted_cols = set()
        self._is_fitted = False

    # ---------------- 공용 사전 저장/불러오기 ----------------
    def save_categories(self, path):
        payload = {'meta': self.categories_meta, 'maps': self.categories}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False)

    def load_categories(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        self.categories_meta = payload['meta']
        self.categories = {k: {self._try_num(vk): vv for vk, vv in v.items()} for k, v in payload['maps'].items()}
        self._is_fitted = True
        self.fitted_cols = set(self.categories.keys())

    @staticmethod
    def _try_num(x):
        # 저장/로드 시 숫자 문자열을 원복
        try:
            return int(x)
        except Exception:
            return x

    # ---------------- 사전 학습 (세 데이터 유니온으로 한 번만) ----------------
    def fit(self, dfs):
        """
        dfs: 리스트 [df1, df2, df3] 처럼 여러 DataFrame
        """
        # 유니온 DF의 카테고리 값 수집
        union_examples = {}
        for df in dfs:
            # 드랍 먼저
            tmp = self.drop_data(df)
            # 카테고리 컬럼만 후보
            cats = return_cols(tmp, 'categorical', boundary=self.categories_meta['boundary'])
            cats = [c for c in cats if c not in self.label_cols]
            for col in cats:
                vals = tmp[col].dropna().unique().tolist()
                union_examples.setdefault(col, set()).update(vals)

        # 각 컬럼별로 1부터 라벨 부여 (NaN은 0)
        for col, vals in union_examples.items():
            # 문자열/숫자 섞여 있을 수 있으니 문자열 캐스팅 기반 정렬 안정화
            # (동일한 입력이면 항상 동일한 순서를 보장)
            sorted_vals = sorted(vals, key=lambda x: (str(type(x)), str(x)))
            start = self.categories_meta['starts_from']  # 1
            self.categories[col] = {val: i for i, val in enumerate(sorted_vals, start=start)}

        self.fitted_cols = set(self.categories.keys())
        self._is_fitted = True
        return self

    # ---------------- 드랍 ----------------
    def set_drop_cols(self, cols):
        self.drop_cols = cols

    def drop_data(self, df):
        to_drop = [c for c in self.drop_cols if c in df.columns]
        return df.drop(columns=to_drop)

    # ---------------- 카테고리 인코딩 (transform 전용) ----------------
    def category_encoding(self, df):
        """
        self.categories(고정 사전)를 사용해 변환.
        NaN -> 0, 사전에 없는 미지 값 -> 새 라벨 부여 X (경고용 -1) 또는 0 중 택1.
        여기서는 일관성 보장을 위해 **미지 값은 -1**로 표기(원하면 0으로 바꿔도 됨).
        """
        if not self._is_fitted:
            raise RuntimeError("category_encoding 전에 fit()을 먼저 호출해 전체 사전을 만들어주세요.")

        df_encoded = df.copy()
        nan_label = self.categories_meta['nan_label']

        # 후보 컬럼
        cats = return_cols(df_encoded, 'categorical', boundary=self.categories_meta['boundary'])
        cats = [c for c in cats if c not in self.label_cols]

        for col in cats:
            if col not in self.categories:
                # fit 시점에 카테고리가 아니었거나 새로 생긴 컬럼 → 그대로 두거나 숫자면 패스
                continue

            label_map = self.categories[col]
            # 매핑
            enc = df_encoded[col].map(label_map)
            # NaN 값(원래 NaN이거나 매핑 실패) 처리
            # 원래 NaN -> 0, 매핑 실패(새로운 값) -> -1
            is_null_original = df_encoded[col].isna()
            enc = enc.astype('Float64')  # NA를 보존하는 타입
            # 매핑 실패 위치: enc.isna() & ~is_null_original
            unknown_mask = enc.isna() & (~is_null_original)

            # 채우기
            enc = enc.fillna(nan_label)  # 우선 전부 0으로
            enc[unknown_mask] = -1       # 미지 카테고리는 -1로 표시 (일관성 강제)

            df_encoded[col] = enc.astype('int64')

        return df_encoded

    # ---------------- 디코딩 ----------------
    def category_decoding(self, df_encoded):
        if self.categories_meta.get('encoding_type') != 'label':
            raise ValueError("self.categories가 'label' 인코딩 정보를 포함하지 않습니다.")
        df_decoded = df_encoded.copy()
        for col, mapping in self.categories.items():
            if col not in df_decoded.columns:
                continue
            inv = {v: k for k, v in mapping.items()}
            # 0(NaN), -1(미지)은 복원 불가 → 그대로 둠
            df_decoded[col] = df_decoded[col].map(inv).where(~df_decoded[col].isin([0, -1]), other=np.nan)
        return df_decoded

    # ---------------- 라벨/시간 생성(골격 유지) ----------------
    def label_data_encoding(self, df):
        """
        TODO:
        - label_Surv_flags 규칙으로 drop/keep
        - COD to site recode 기준으로 사건별 타깃 라벨(-1,0,1,2,3) 생성 -> 'target_label'
        - 'Survival months'를 3개월 단위로 binning -> 'Survival months_bin_3m'
        - 사용한 원천 컬럼 drop
        """
        df_label = df.copy()
        # ---- 여기서 구현 ----
        return df_label

    # ---------------- 파이프라인 ----------------
    def transform(self, df):
        df_cleaned = self.drop_data(df)
        df_cleaned = self.category_encoding(df_cleaned)
        df_cleaned = self.label_data_encoding(df_cleaned)
        return df_cleaned

    def run(self, df):
        # run은 transform 별칭 (호출 습관 유지용)
        return self.transform(df)
    
class DataPreprocessing() :
    # 라벨을 구성하기 위한 사인별 카테고리 설정 -> COD to site recode
    label_cod_list = [
            # 1. Cancer (암)
            [
                "Lung and Bronchus","Colon excluding Rectum","Rectum and Rectosigmoid Junction",
                "Liver","Intrahepatic Bile Duct","Stomach","Esophagus","Pancreas","Breast",
                "Prostate","Urinary Bladder","Kidney and Renal Pelvis","Ovary","Corpus Uteri",
                "Cervix Uteri","Uterus, NOS","Vulva","Vagina","Testis","Penis","Thyroid",
                "Other Endocrine including Thymus","Brain and Other Nervous System","Eye and Orbit",
                "Bones and Joints","Soft Tissue including Heart","Peritoneum, Omentum and Mesentery",
                "Retroperitoneum","Other Urinary Organs","Gallbladder","Other Biliary",
                "Small Intestine","Other Digestive Organs","Trachea, Mediastinum and Other Respiratory Organs",
                "Nose, Nasal Cavity and Middle Ear","Tongue","Floor of Mouth","Gum and Other Mouth",
                "Lip","Tonsil","Oropharynx","Nasopharynx","Hypopharynx","Other Oral Cavity and Pharynx",
                "Non-Hodgkin Lymphoma","Hodgkin Lymphoma","Myeloma","Acute Myeloid Leukemia",
                "Chronic Lymphocytic Leukemia","Chronic Myeloid Leukemia","Acute Lymphocytic Leukemia",
                "Other Myeloid/Monocytic Leukemia","Other Acute Leukemia","Other Lymphocytic Leukemia",
                "Melanoma of the Skin","Non-Melanoma Skin","Miscellaneous Malignant Cancer",
                "In situ, benign or unknown behavior neoplasm"
            ],

            # 2. Complications (합병증)
            [
                "Septicemia","Pneumonia and Influenza",
                "Other Infectious and Parasitic Diseases including HIV",
                "Chronic Liver Disease and Cirrhosis",
                "Nephritis, Nephrotic Syndrome and Nephrosis",
                "Chronic Obstructive Pulmonary Disease and Allied Cond",
                "Diabetes Mellitus","Hypertension without Heart Disease",
                "Alzheimers (ICD-9 and 10 only)"
            ],

            # 3. Other Non-cancer (질병계열, 병원 개입 한계 큰 원인 포함)
            [
                "Diseases of Heart","Cerebrovascular Diseases","Aortic Aneurysm and Dissection",
                "Other Diseases of Arteries, Arterioles, Capillaries","Atherosclerosis",
                "Stomach and Duodenal Ulcers","Congenital Anomalies",
                "Complications of Pregnancy, Childbirth, Puerperium"
            ],

            # 4. External / Unclear (외인사·불명)
            [
                "Accidents and Adverse Effects",
                "Homicide and Legal Intervention",
                "State DC not available or state DC available but no COD",
                "Symptoms, Signs and Ill-Defined Conditions",
                "Other Cause of Death"
            ],

            # 5. Suicide
            [
                "Suicide and Self-Inflicted Injury"
            ]
        ]

    # 라벨을 구성하기 위한 관측 타입별 카테고리 설정 -> Survival months flag
    label_Surv_flags = [
            # 1. 문제 없는 정상 데이터
            [
                "Complete dates are available and there are more than 0 days of survival"
            ],

            # 2. 부정확한 데이터 (Alive -> 그대로 사용, Dead -> drop)
            [
                "Incomplete dates are available and there cannot be zero days of follow-up"
            ],

            # 3. 부정확한 데이터 -> drop
            [
                "Not calculated because a Death Certificate Only or Autopsy Only case",
                "Complete dates are available and there are 0 days of survival",
                "Incomplete dates are available and there could be zero days of follow-up"
            ]
        ]
    
    def __init__(self, drop_cols=None, label_cols=None) :
        self.categories = {}    # 카테고리의 인코딩을 저장

        ###--- 드랍하기로 한 컬럼명을 작성
        self.drop_cols = ['Patient ID','Year of follow-up recode','Site recode ICD-O-3/WHO 2008','Number of Cores Positive Recode (2010+)',
                          'Number of Cores Examined Recode (2010+)','Vital status recode (study cutoff used)']     # 드랍할 컬럼명을 저장
        if drop_cols is not None :
            self.drop_cols = drop_cols

        ###--- 라벨로 사용할 데이터와, 시간 데이터의 컬럼명 작성
        self.label_cols = ['Survival months', 'COD to site recode', 'Survival months flag']
        if label_cols is not None :
            self.label_cols = label_cols

    # 드랍할 컬럼을 설정
    def set_drop_cols(self, cols):
        self.drop_cols = cols

    # 데이터 드랍
    def drop_data(self, df) :
        return df.drop(columns=self.drop_cols)

    # 범주형 데이터 라벨 인코딩
    def category_encoding(self, df):
        # 범주형 데이터의 컬럼명만 반환
        categorical_col = return_cols(df, 'categorical', 100)

        # exclude_cols에 포함된 컬럼은 인코딩 대상에서 제거
        categorical_col = [c for c in categorical_col if c not in self.label_cols]

        df_encoded = df.copy()

        # 인코딩 타입 기록
        self.categories['encoding_type'] = 'label'

        for col in categorical_col:
            # 기존 인코딩 규칙이 존재하지 않는 경우 새로 생성
            if col not in self.categories:
                unique_vals = sorted(df_encoded[col].dropna().unique())  # NaN 제외
                label = {val: i for i, val in enumerate(unique_vals)}
                self.categories[col] = label

            label_map = self.categories[col]

            # 새로운 값이 있으면 기존 라벨에 추가
            new_vals = sorted(set(df_encoded[col].dropna()) - set(label_map.keys()))
            if new_vals:
                start_idx = len(label_map)
                for i, val in enumerate(new_vals, start=start_idx):
                    label_map[val] = i
                self.categories[col] = label_map  # 업데이트 반영

            # 매핑 수행
            df_encoded[col] = df_encoded[col].map(label_map)

            # 매핑되지 않은 값(NaN, NULL 등)은 -1로 처리
            df_encoded[col] = df_encoded[col].fillna(-1).astype(int)

        return df_encoded

    # 저장된 딕셔너리를 통해 디코딩
    def category_decoding(self, df_encoded):
        """
        인코딩된 데이터프레임을 self.categories 딕셔너리를 사용하여 복원하는 함수.
        category_encoding()에서 사용된 라벨 인코딩을 역변환함.

        Args:
            df_encoded (pd.DataFrame): 인코딩된 데이터프레임

        Returns:
            pd.DataFrame: 복원된 데이터프레임
        """

        if 'encoding_type' not in self.categories or self.categories['encoding_type'] != 'label':
            raise ValueError("self.categories가 'label' 인코딩 정보를 포함하지 않습니다.")

        df_decoded = df_encoded.copy()

        for col, mapping in self.categories.items():
            if col == 'encoding_type':
                continue
            if col not in df_decoded.columns:
                continue

            # 역매핑 딕셔너리 생성
            inverse_map = {v: k for k, v in mapping.items()}
            df_decoded[col] = df_decoded[col].map(inverse_map)

        return df_decoded
    
    # 라벨 데이터 인코딩
    def label_data_encoding(self, df):
        df_label = df.copy()

        # 사용될 주요 컬럼명
        cod_col = 'COD to site recode'
        surv_flag_col = 'Survival months flag'
        surv_months_col = 'Survival months'

        # 라벨링 규칙 정의 (이미 선언된 리스트 이용)
        label_cod_list = self.label_cod_list
        label_Surv_flags = self.label_Surv_flags

        unusable_flags = set(label_Surv_flags[2])  # 드랍 대상
        drop_idx = df_label[df_label[surv_flag_col].isin(unusable_flags)].index
        df_label = df_label.drop(index=drop_idx).reset_index(drop=True)

        df_label[surv_months_col] = pd.to_numeric(df_label[surv_months_col], errors='coerce')
        surv_bin = (df_label[surv_months_col] // 3).astype(int)
        df_label['Survival months_bin_3m'] = surv_bin.clip(lower=0, upper=90)

        target_labels = []

        for _, row in df_label.iterrows():
            cod = row[cod_col]
            surv_flag = row[surv_flag_col]

            # 생존인 경우 (COD가 생존으로 표시된 케이스)
            if cod in [None, np.nan, '', 'Alive']:
                target_labels.append(-1)
                continue

            # COD에 따라 범주 결정
            label_value = None
            for i, cod_group in enumerate(label_cod_list):
                if cod in cod_group:
                    if i == 3 :
                        label_value = -1
                    elif i == 4 :
                        label_value = 3
                    else :
                        label_value = i  # 0~4 중 하나
                    break

            # 라벨이 매칭되지 않은 경우 → 기타(-1 처리 대신 무시)
            if label_value is None:
                label_value = -1

            target_labels.append(label_value)

        df_label['target_label'] = target_labels

        df_label = df_label.drop(columns=[cod_col, surv_flag_col, surv_months_col])

        return df_label


    def run(self, df) :
        df_cleaned = self.drop_data(df)
        df_cleaned = self.category_encoding(df_cleaned)
        df_cleaned = self.label_data_encoding(df_cleaned)

        return df_cleaned


# 라벨을 구성하기 위한 사인별 카테고리 설정 -> COD to site recode
label_cod_list = [
        # 1. Cancer (암)
        [
            "Lung and Bronchus","Colon excluding Rectum","Rectum and Rectosigmoid Junction",
            "Liver","Intrahepatic Bile Duct","Stomach","Esophagus","Pancreas","Breast",
            "Prostate","Urinary Bladder","Kidney and Renal Pelvis","Ovary","Corpus Uteri",
            "Cervix Uteri","Uterus, NOS","Vulva","Vagina","Testis","Penis","Thyroid",
            "Other Endocrine including Thymus","Brain and Other Nervous System","Eye and Orbit",
            "Bones and Joints","Soft Tissue including Heart","Peritoneum, Omentum and Mesentery",
            "Retroperitoneum","Other Urinary Organs","Gallbladder","Other Biliary",
            "Small Intestine","Other Digestive Organs","Trachea, Mediastinum and Other Respiratory Organs",
            "Nose, Nasal Cavity and Middle Ear","Tongue","Floor of Mouth","Gum and Other Mouth",
            "Lip","Tonsil","Oropharynx","Nasopharynx","Hypopharynx","Other Oral Cavity and Pharynx",
            "Non-Hodgkin Lymphoma","Hodgkin Lymphoma","Myeloma","Acute Myeloid Leukemia",
            "Chronic Lymphocytic Leukemia","Chronic Myeloid Leukemia","Acute Lymphocytic Leukemia",
            "Other Myeloid/Monocytic Leukemia","Other Acute Leukemia","Other Lymphocytic Leukemia",
            "Melanoma of the Skin","Non-Melanoma Skin","Miscellaneous Malignant Cancer",
            "In situ, benign or unknown behavior neoplasm"
        ],

        # 2. Complications (합병증)
        [
            "Septicemia","Pneumonia and Influenza",
            "Other Infectious and Parasitic Diseases including HIV",
            "Chronic Liver Disease and Cirrhosis",
            "Nephritis, Nephrotic Syndrome and Nephrosis",
            "Chronic Obstructive Pulmonary Disease and Allied Cond",
            "Diabetes Mellitus","Hypertension without Heart Disease",
            "Alzheimers (ICD-9 and 10 only)"
        ],

        # 3. Other Non-cancer (질병계열, 병원 개입 한계 큰 원인 포함)
        [
            "Diseases of Heart","Cerebrovascular Diseases","Aortic Aneurysm and Dissection",
            "Other Diseases of Arteries, Arterioles, Capillaries","Atherosclerosis",
            "Stomach and Duodenal Ulcers","Congenital Anomalies",
            "Complications of Pregnancy, Childbirth, Puerperium"
        ],

        # 4. External / Unclear (외인사·불명)
        [
            "Accidents and Adverse Effects",
            "Homicide and Legal Intervention",
            "State DC not available or state DC available but no COD",
            "Symptoms, Signs and Ill-Defined Conditions",
            "Other Cause of Death"
        ],

        # 5. Suicide
        [
            "Suicide and Self-Inflicted Injury"
        ]
    ]

# 라벨을 구성하기 위한 관측 타입별 카테고리 설정 -> Survival months flag
label_Surv_flags = [
        # 1. 문제 없는 정상 데이터
        [
            "Complete dates are available and there are more than 0 days of survival"
        ],

        # 2. 부정확한 데이터 (Alive -> 그대로 사용, Dead -> drop)
        [
            "Incomplete dates are available and there cannot be zero days of follow-up"
        ],

        # 3. 부정확한 데이터 -> drop
        [
            "Not calculated because a Death Certificate Only or Autopsy Only case",
            "Complete dates are available and there are 0 days of survival",
            "Incomplete dates are available and there could be zero days of follow-up"
        ]
    ]
 
# 데이터 프레임에서, boundary를 기준으로 연속형, 혹은 범주형에 속하는 컬럼명을 반환
def return_cols(df, type, boundary=100) :    # type : ('continuous', 'categorical'), boundary : 두 분류를 결정할 서로 다른 요소의 수
    cols = []
    if type == 'continuous' :
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique(dropna=True) >= boundary :
                cols.append(col)

    elif type == 'categorical' :
        for col in df.columns:
            if df[col].nunique(dropna=True) < boundary :
                cols.append(col)
    else :
        ValueError('Wrong type.')

    return cols