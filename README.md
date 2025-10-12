# **💉 암 환자 고위험군 및 사망률 예측**
---

## 1. **팀 소개**
### 🚑  응급디버깅실 (ER) 🚑
코드 터지면 바로 실려옴 💻

## 🧬 팀원 소개 
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

1. 암 환자에 대한 시간에 따른 연속적인 생존율 예측
   암 환자의 주요 사망 원인별 사망률을 예측하여 환자에 대한 예상 생존 기간을 예측
2. 생존율 예측을 기반으로한 위험 점수 계산 및 예측
   예측한 생존율을 기반으로 각 환자별 위험 정도를 0~100 사이의 점수로 계산하여 고위험군 환자를 한눈에 파악할 수 있도록 함


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

### 6-1. 기본 모델 아이디어
 해당 시스템은 환자에 대한 사망원인별 연속적인 사망률을 예측하여, 고위험 요인에 대한 선제적 조치와 예방을 목적으로 한다.
 따라서 여러 시간대별 사망률을 예측할 수 있는 모델을 필요로 하고, 해당 형태를 구현하기 위하여 딥러닝 모델을 구현하여 사용하였다.
 기본 모델의 형태와 손실함수는 *DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks.* (Lee, Changhee, et al., 2018)를 기반으로 작성하였다.

 <div align="center">
  <img src="" alt="Deephit original" style="max-width: 100%; height: auto; margin-bottom: 20px;"/>
  <br>
  <i> Deephit 모델의 기본 구조 </i>
</div>

  DeepHit모델은 특성을 공유 Branch와 각 사건별 branch에 차례대로 통과시켜 이산화시킨 시간 별 사건 발생 확률을 예측하는 형태의 모델이다.

### 6-2. 모델 개선 아이디어
  Deephit 모델은 시간대별 사건 발생 확률을 예측하는 MLP 기반의 모델로, 학습 과정에 여러 모듈을 추가하여 성능 개선을 꾀할 수 있다.

#### 6-2.1. SEBlock (Squeeze-and-Excitation Block)
  Standard Scaler를 이용한 스케일링 대신 모델에 *Squeeze-and-Excitation Networks* (Jie Hu, Li Shen, Gang Sun, 2018) 에서 사용된 SEBlock 아이디어를 단순 특성 MLP에 적용하여 Feature Weighting

#### 6-2.2. Residual Connection, Feature-wise Concat
  모델의 학습을 돕고 성능을 향상시키기 위하여 사용
   
  - 모델의 깊이가 깊지 않아 늘어난 계산 복잡도에 비하여 성능 향상이 매우 미미하여 최종 모델에서는 사용하지 않음

#### 6-2.3. 1D, 2D CNN
  모델의 결과에 시간대별, 사건별 연관성을 추가하기 위하여 CNN을 사용

  - 눈에 띄는 성능 향상을 보이지 않음

#### -모델별 학습 결과



### 6-3. 최종 모델 형태

<div align="center">
  <img src="" alt="Deephit SEBlock" style="max-width: 100%; height: auto; margin-bottom: 20px;"/>
  <br>
  <i> 최종 Deephit 모델 </i>
</div>


## 7. **수행 결과**
---

### 결론
---

## 8. **한줄 회고**

<table>
  <tr>
    <th style="width:80px;">이름</th>
    <th>회고 내용</th>
  </tr>
  <tr>
    <td>🧠 박준영</td>
    <td></td>
  </tr>
  <tr>
    <td>🌡️ 강지완</td>
    <td></td>
  </tr>
  <tr>
    <td>👩🏻‍⚕️ 김민정 </td>
    <td>
</td>
  </tr>
  <tr>
    <td>🧑🏻‍⚕️ 이승원</td>
    <td>
    </td>
  </tr>
  <tr>
    <td>🫀 박소희</td>
    <td></td>
  </tr>
</table>












