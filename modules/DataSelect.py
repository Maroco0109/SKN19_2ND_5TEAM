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
    label_cod_list에서 5번 라벨(사고사)에 해당 : -1
    암 관련으로 사망 : 0
    합병증 관련으로 사망 : 1
    기타 질병으로 사망 : 2
    자해 및 자살로 인한 사망 : 3


"""

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
def return_cols(df, type, boundary=15) :    # type : ('continuous', 'categorical'), boundary : 두 분류를 결정할 서로 다른 요소의 수
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