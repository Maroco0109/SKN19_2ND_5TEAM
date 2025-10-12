# **암 환자 고위험군 및 사망률 예측 💉**
---

## 1. **팀 소개**
## 5팀

### 🧑‍⚕️팀원 소개
---
<table>
  <tr>
    <td align="center" width="200px">
      <img src="" width="120px" height="120px" alt="박준영" style="border-radius: 50%;"><br/>
      <b>박준영</b>
    </td>
    <td align="center" width="200px">
      <img src="" width="120px" height="120px" alt="강지완" style="border-radius: 50%;"><br/>
      <b>강지완</b>
    </td>
    <td align="center" width="200px">
      <img src="" width="120px" height="120px" alt="김민정" style="border-radius: 50%;"><br/>
      <b>김민정</b>
    </td>
    <td align="center" width="200px">
      <img src="" width="120px" height="120px" alt="이승원" style="border-radius: 50%;"><br/>
      <b>이승원</b>
    </td>
   <td align="center" width="200px">
      <img src="" width="120px" height="120px" alt="박소희" style="border-radius: 50%;"><br/>
      <b>박소희</b>
    </td>
  </tr>
  <tr>
    <td align="center" width="200px">
      <a href="https://github.com/deneb784"> GitHub</a>
    </td>
    <td align="center" width="200px">
      <a href="https://github.com/Maroco0109"> GitHub</a>
    </td>
    <td align="center" width="200px">
      <a href=https://github.com/MinJeung-Kim"> GitHub</a>
    </td>
    <td align="center" width="200px">
      <a href="https://github.com/seungwon-sw"> GitHub</a>
    </td>
     <td align="center" width="200px">
      <a href="https://github.com/xxoysauce"> GitHub</a>
    </td>
  </tr>
  
  
</table>


---

## 2. **프로젝트 개요**
---
### 2.1. 프로젝트 명
#### **암 환자 고위험군 및 이탈율 예측**

### 2.2. 프로젝트 주제 선정 배경
질병 환자의 사망은 병원의 비즈니스적인 입장에서 생각하면 고객의 이탈과 동일하다고 볼 수 있다. 또한 환자의 완치율이 높은 병원은 꾸준히 내원하고 방문하는 사람들이 많은데 반해 사망률이 높은 병원은 내원하던 환자들도 다른 병원으로 옮기는 등 방문하는 사람이 적어지듯이, 환자의 사망률은 고객 유치 측면과 기존 고객의 유지와 직접적인 관련이 있다.  따라서 고위험군 환자를 미리 식별하고, 적절한 선제적 조치를 취하는 것은 병원의 관리 전략 측면에서 매우 중요하다. 이러한 접근은 단순히 통계적인 질병 분석을 넘어, 병원의 서비스 지속성을 확보하는 고객 관리 전략의 일부라고 할 수 있다.

### 2.3. 프로젝트 목적
본 프로젝트의 목적은 다음과 같다.

1. 알나린ㄹㅇ늘
2. ㅇㄴ라ㅣㄴ라ㅣ


---
## 3-1. **기술 스택**


|      **카테고리**     |                                                                                                                                                                                                                    **기술 스택**                                                                                                                                                                                                                    |
| :---------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|      **WEB**      |                                                                                                                                                 <img src="https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" width="120"/>                                                                                                                                                |
|     **라이브러리**     |     <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" width="120"/> <img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black" alt="matplotlib" width="120" /> <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="Pytorch" width="120" />    |
| **개발 환경 및 협업 도구** | <img src="https://img.shields.io/badge/GitBook-%23000000.svg?style=for-the-badge&logo=gitbook&logoColor=white" alt="Git" width="120"/> <img src="https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white" alt="VSCode" width="120"/> <img src="https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white" alt="Notion" width="120"/> |


## 3-2. **파일 구조**

```
TEAMPROJECT/
└─ SKN19_2ND_5TEAM/
   ├─ data/
   │  ├─ parameters/
   │  │   ├─ deephit_model_feature_2dcnn.pth
   │  │   ├─ deephit_model_feature_100time.pth
   │  │   ├─ deephit_model_feature_cnn.pth
   │  │   ├─ deephit_model_feature_concat.pth
   │  │   ├─ deephit_model_feature_SE.pth
   │  │   └─ deephit_model_without_feature_concat.pth
   │  │
   │  ├─ 2022Data_part1.csv
   │  ├─ 2022Data_part2.csv
   │  ├─ categories_select.csv
   │  ├─ encoded_dataset_COD.csv
   │  ├─ encoded_dataset.csv
   │  ├─ Suicide_encode.csv
   │  ├─ Suicide.csv
   │  └─ test dataset.csv
   │
   ├─ insight/
   │  ├─ COD list.ipynb
   │  ├─ data_encode_insight_kmj.ipynb
   │  ├─ data_insight_kjw.ipynb
   │  ├─ data_insight_kmj.ipynb
   │  ├─ data_insight_lsw.ipynb
   │  ├─ EDA.ipynb
   │  └─ encoded_label_dump.txt
   │
   ├─ modules/
   │  ├─ __init__.py
   │  ├─ DataAnalysis.py
   │  ├─ DataModify.py
   │  ├─ DataSelect.py
   │  ├─ ModelAnalysis.py
   │  ├─ Models.py
   │  └─ smart_csv.py
   │
   ├─ parameters/
   │  ├─ categories.pkl
   │  └─ deephit_model_feature.pth
   │
   ├─ .gitignore
   ├─ Analysis.ipynb
   ├─ environment.yml
   ├─ pyproject.toml
   ├─ README.md
   ├─ run.py
   ├─ test.ipynb
   └─ train.ipynb

```

## 4. **WBS**
---

## 5. **데이터 전처리 및 EDA**
___


## 6. **인공지능 학습 결과서**
---
### -모델별 학습 결과


## 7. **수행 결과**
---

### 결론
---

## 8. **한줄 회고**












