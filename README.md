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
      <img src="./public/1.png" width="120px" height="120px" alt="박준영" style="border-radius: 50%;"><br/>
      <b>박준영</b>
    </td>
    <td align="center" width="200px">
      <img src="./public/3.png" width="120px" height="120px" alt="강지완" style="border-radius: 50%;"><br/>
      <b>강지완</b>
    </td>
    <td align="center" width="200px">
      <img src="./public/2.png" width="120px" height="120px" alt="김민정" style="border-radius: 50%;"><br/>
      <b>김민정</b>
    </td>
    <td align="center" width="200px">
      <img src="./public/4.png" width="120px" height="120px" alt="이승원" style="border-radius: 50%;"><br/>
      <b>이승원</b>
    </td>
   <td align="center" width="200px">
      <img src="./public/5.png" width="120px" height="120px" alt="박소희" style="border-radius: 50%;"><br/>
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
      <a href="https://github.com/MinJeung-Kim"> GitHub</a>
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
### **암 환자 고위험군 및 이탈율 예측**

### 2.2. 프로젝트 주제 선정 배경
> 병원 관점에서 질병 환자의 **사망**은 단순한 의료적 사건이 아니라, **고객의 이탈(churn)** 과 유사한 개념으로 볼 수 있다.


- 완치율이 높은 병원은 환자의 신뢰를 얻어 재방문율이 높고, 새로운 환자 유입도 활발하다.
- 반면 사망률이 높은 병원은 환자의 신뢰 하락으로 인해 내원율이 감소하고, 기존 환자들이 다른 병원으로 이동하는 등 **고객 유지율(환자 유지율)** 이 떨어지는 경향을 보인다.

따라서 병원 경영 측면에서는 **고위험군 환자를 조기에 식별**하고, 치료 개입이나 관리 강화를 통해 위험을 완화하는 것이 매우 중요하다.
이는 단순한 질병 통계 분석을 넘어, 병원의 **서비스 지속성과 환자 관리 효율성**을 높이는 핵심 전략으로 기능한다.

### 2.3. 프로젝트 목적
암 환자에 대한 관찰과 추적은 지속적으로 이루어져야한다. 특정 기간에 대한 생존율을 예측하기에는 장기 치료가 불가피한 경우도 많기에, 다양한 기간에서의 생존율을 예측할 필요가 있다.    
따라서 해당 프로젝트에서는 각 시간대별, 사망 원인별 환자의 생존율을 추정하여 각 환자에 맞는 관리 전략을 제공하는데 도움을 주고자 한다.

1. 암 환자의 시간에 따른 생존율 예측

- 생존 분석(Survival Analysis)을 통해 환자의 경과 시간에 따른 생존 확률을 추정한다.

2. 주요 사망 원인별 사망률 예측 및 생존 기간 산출

- 암 유형별·사망 원인별로 사망 확률을 분석하여 환자의 예상 생존 기간을 예측한다.

3. 예측 기반 위험 점수(Risk Score) 산출

- 예측된 생존율을 바탕으로 환자별 위험도를 0~100점 범위로 정량화하고,
고위험군 환자를 직관적으로 식별할 수 있는 관리 지표를 제공한다.

4. 생존 인원들의 치료 중단 시기 예측
- 예측된 생존율을 바탕으로 생존이 예측되는 인원들 역시 치료 중단 시기를 예측한다.


---
## 3-1. **기술 스택**


| **카테고리** | **기술스택** |
| :--- | :--- |
| **WEB** | <img src="https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" width="120"/> |
| **라이브러리** | <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" width="120"/> <img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black" alt="matplotlib" width="120"/> <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="Pytorch" width="120"/> <img src="https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly" width="120"/> <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="Numpy" width="120"/> |
| **개발 환경 및 협업 도구** | <img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white" alt="Git" width="120"/> <img src="https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white" alt="VSCode" width="120"/> <img src="https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white" alt="Notion" width="120"/> |


## 3-2. **파일 구조**

```
TEAMPROJECT/
└─ SKN19_2ND_5TEAM/
   ├─ data/
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
   │  ├─ deephit_model_2D_CNN.pth
   │  ├─ deephit_model_CNN.pth
   │  ├─ deephit_model_Concat.pth
   │  ├─ deephit_model_feature.pth
   │  └─ deephit_model_feature_original_Deephit.pth 
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
    
    
<html lang="ko"><head>
  <meta charset="UTF-8">
  <title>WBS Table</title>
</head>
<body>
<table>
  <thead>
    <tr>
      <th align="right"></th>
      <th align="left">WBS Number</th>
      <th align="left">Task Title</th>
      <th align="left">담당자</th>
      <th align="left">시작일</th>
      <th align="left">종료일</th>
      <th align="left">산출물</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="right"></td>
      <td align="left">1</td>
      <td align="left">프로젝트 컨셉트 및 방향성</td>
      <td align="left">강지완, 김민정, 박소희, 박준영, 이승원</td>
      <td align="left">2025년 9월 30일</td>
      <td align="left">2025년 10월 5일</td>
      <td align="left">2022Data_part1.csv,
2022Data_part2.csv</td>
    </tr>
    <tr>
      <td align="right"></td>
      <td align="left">1.1.</td>
      <td align="left">프로젝트 주제 선정</td>
      <td align="left">강지완, 김민정, 박소희, 박준영, 이승원</td>
      <td align="left">2025년 9월 30일</td>
      <td align="left">2025년 10월 2일</td>
      <td align="left"></td>
    </tr>
    <tr>
      <td align="right"></td>
      <td align="left">1.2</td>
      <td align="left">데이터셋 분석</td>
      <td align="left">강지완, 김민정, 박소희, 박준영, 이승원</td>
      <td align="left">2025년 10월 1일</td>
      <td align="left">2025년 10월 5일</td>
      <td align="left"></td>
    </tr>
    <tr>
      <td align="right"></td>
      <td align="left">1.2.1</td>
      <td align="left">타겟 레이블 분석</td>
      <td align="left">강지완, 김민정, 박소희, 박준영, 이승원</td>
      <td align="left">2025년 10월 1일</td>
      <td align="left">2025년 10월 2일</td>
      <td align="left"></td>
    </tr>
    <tr>
      <td align="right"></td>
      <td align="left">1.2.2</td>
      <td align="left">컬럼명 분석</td>
      <td align="left">강지완, 김민정, 박소희, 박준영, 이승원</td>
      <td align="left">2025년 10월 1일</td>
      <td align="left">2025년 10월 5일</td>
      <td align="left"></td>
    </tr>
    <tr>
      <td align="right"></td>
      <td align="left">1.2.3</td>
      <td align="left">피처값 분석 (컬럼 내용)</td>
      <td align="left">강지완, 김민정, 박소희, 박준영, 이승원</td>
      <td align="left">2025년 10월 1일</td>
      <td align="left">2025년 10월 5일</td>
      <td align="left"></td>
    </tr>
    <tr>
      <td align="right"></td>
      <td align="left">2</td>
      <td align="left">EDA</td>
      <td align="left">강지완, 김민정, 이승원</td>
      <td align="left">2025년 10월 2일</td>
      <td align="left">2025년 10월 10일</td>
      <td align="left">encoded_dataset.csv<br>encoded_dataset_COD.csv</td>
    </tr>
    <tr>
      <td align="right"></td>
      <td align="left">2.1</td>
      <td align="left">데이터셋 인코딩</td>
      <td align="left">강지완, 박소희, 박준영</td>
      <td align="left">2025년 10월 2일</td>
      <td align="left">2025년 10월 3일</td>
      <td align="left">Analysis.ipynb<br>DataModify.ipynb<br>DataSelect.py</td>
    </tr>
    <tr>
      <td align="right"></td>
      <td align="left">2.1</td>
      <td align="left">데이터 기본 정보 분석</td>
      <td align="left">강지완, 김민정, 이승원</td>
      <td align="left">2025년 10월 10일</td>
      <td align="left">2025년 10월 12일</td>
      <td align="left">data_insight_kjw.ipynb<br>data_insight_kmj.ipynb<br>data_insight_lsw.ipynb</td>
    </tr>
    <tr>
      <td align="right"></td>
      <td align="left">2.2</td>
      <td align="left">사망 원인 기반 정보 분석</td>
      <td align="left">강지완, 김민정, 이승원</td>
      <td align="left">2025년 10월 10일</td>
      <td align="left">2025년 10월 14일</td>
      <td align="left">data_insight_kjw.ipynb<br>data_insight_kmj.ipynb<br>data_insight_lsw.ipynb</td>
    </tr>
    <tr>
      <td align="right"></td>
      <td align="left">2.3</td>
      <td align="left">타겟 레이블 상관관계 분석</td>
      <td align="left">강지완, 김민정, 이승원</td>
      <td align="left">2025년 10월 10일</td>
      <td align="left">2025년 10월 14일</td>
      <td align="left">EDA.ipynb<br>DataModify.ipynb<br>DataAnalysis.ipynb</td>
    </tr>
    <tr>
      <td align="right"></td>
      <td align="left">3</td>
      <td align="left">Modeling</td>
      <td align="left">박소희, 박준영</td>
      <td align="left">2025년 10월 2일</td>
      <td align="left">2025년 10월 15일</td>
      <td align="left"></td>
    </tr>
    <tr>
      <td align="right"></td>
      <td align="left">3.1</td>
      <td align="left">기본 모델 구현</td>
      <td align="left">박준영</td>
      <td align="left">2025년 10월 2일</td>
      <td align="left">2025년 10월 15일</td>
      <td align="left">Models.py<br>ModelAnalysis.py<br>train.ipynb<br>test.ipynb<br>accuraccy_test.ipynb<br>code_test.ipynb</td>
    </tr>
    <tr>
      <td align="right"></td>
      <td align="left">3.2</td>
      <td align="left">모듈 추가 및 각 모델 학습</td>
      <td align="left">박소희, 박준영</td>
      <td align="left">2025년 10월 6일</td>
      <td align="left">2025년 10월 12일</td>
      <td align="left">deephit_model_2D_CNN.pth<br>deephit_model_feature_2dcnn.pth<br>risk_model_event_linears.pth</td>
    </tr>
    <tr>
      <td align="right"></td>
      <td align="left">3.3</td>
      <td align="left">모델 구조 시각화</td>
      <td align="left">박준영, 박소희</td>
      <td align="left">2025년 10월 10일</td>
      <td align="left">2025년 10월 12일</td>
      <td align="left">Deephit CNN.png<br>Deephit_original.png<br>model2.png<br>original model.png</td>
    </tr>
    <tr>
      <td align="right"></td>
      <td align="left">4</td>
      <td align="left">README 정리</td>
      <td align="left">강지완, 김민정, 이승원, 박소희, 박준영</td>
      <td align="left">2025년 10월 11일</td>
      <td align="left">2025년 10월 15일</td>
      <td align="left">README.md</td>
    </tr>
    <tr>
      <td align="right"></td>
      <td align="left">4.1</td>
      <td align="left">README 초안 작성</td>
      <td align="left">박소희</td>
      <td align="left">2025년 10월 11일</td>
      <td align="left">2025년 10월 13일</td>
      <td align="left"></td>
    </tr>
    <tr>
      <td align="right"></td>
      <td align="left">4.2</td>
      <td align="left">EDA 및 시각화 정리</td>
      <td align="left">김민정</td>
      <td align="left">2025년 10월 14일</td>
      <td align="left">2025년 10월 14일</td>
      <td align="left"></td>
    </tr>
    <tr>
      <td align="right"></td>
      <td align="left">4.3</td>
      <td align="left">데이터 전처리 방법 정리</td>
      <td align="left">강지완</td>
      <td align="left">2025년 10월 14일</td>
      <td align="left">2025년 10월 14일</td>
      <td align="left"></td>
    </tr>
    <tr>
      <td align="right"></td>
      <td align="left">4.4</td>
      <td align="left">모델 학습 결과 정리</td>
      <td align="left">박준영, 박소희</td>
      <td align="left">2025년 10월 12일</td>
      <td align="left">2025년 10월 15일</td>
      <td align="left"></td>
    </tr>
    <tr>
      <td align="right"></td>
      <td align="left">5</td>
      <td align="left">Streamlit 구현</td>
      <td align="left">김민정, 박소희, 박준영, 이승원</td>
      <td align="left">2025년 10월 13일</td>
      <td align="left">2025년 10월 15일</td>
      <td align="left">run.py</td>
    </tr>
    <tr>
      <td align="right"></td>
      <td align="left">5.1</td>
      <td align="left">Streamlit 초안 작성</td>
      <td align="left">박준영</td>
      <td align="left">2025년 10월 13일</td>
      <td align="left">2025년 10월 13일</td>
      <td align="left"></td>
    </tr>
    <tr>
      <td align="right"></td>
      <td align="left">5.2</td>
      <td align="left">전체 UI 구성 및 차트 디자인 변경</td>
      <td align="left">김민정, 박소희, 이승원</td>
      <td align="left">2025년 10월 14일</td>
      <td align="left">2025년 10월 15일</td>
      <td align="left"></td>
    </tr>
    <tr>
      <td align="right"></td>
      <td align="left">5.3</td>
      <td align="left">UI 병합 및 내용 보충</td>
      <td align="left">박준영, 박소희</td>
      <td align="left">2025년 10월 14일</td>
      <td align="left">2025년 10월 15일</td>
      <td align="left"></td>
    </tr>
  </tbody>
</table>


</body></html>
---

## 5. **데이터 전처리 및 EDA**
   
모델의 학습과 평가에 사용한 데이터는 미국국립암연구소(NCI)에서 제공하는 암 환자에 대한 데이터셋인 SEER(Surveillance, Epidemiology, and End Results) 데이터셋의 일부를 사용하였다.   
2022년에 내원한 기록이 있는 주요 4대 암(폐, 간, 위, 대장) 환자에 대한 데이터 약 500,000개를 기반으로 해당 과정을 진행하였다.
   
### [데이터 전처리]
```modules/DataModify.py```
---------------------
```python
# DataPreprocessing
├─ category_encoding()                # 범주형 컬럼을 라벨/원-핫 인코딩
├─ decode_csv_features()              # 범주형 변수를 모델에 적용할 수 있도록 디코딩
├─ encode_ordinal_columns()           # 정의된 순서(또는 수치 그대로 유지)를 적용해 순서형 변수 일관성을 유지
├─ encode_nominal_columns()           # 명목형 변수(순서가 없는 범주형 변수) 전체를 새 ID로 팩터라이즈하고 결측은 -1로 통일
├─ create_combined_label()            # 생존 상태·사망원인을 조합해 `target_label` 다중 클래스를 구축
├─ bin_survival_months()              # 생존 개월을 3개월 단위 등 균일 구간으로 나눠 시간 기반 특징 강화
└─ _normalize_seer_summary_stage()    # 확장 병기 코드를 0/1/2/3/9 표준 범주로 정규화
```
---
### [ 데이터 기본 정보 분석 ] 
  <img src="./insight/img/plot_basic_distributions.png" width="100%" />
  

### [ 사망 원인 및 상관관계 분석 ] 
  <img src="./insight/img/plot_cod_top_and_age_pattern.png" width="100%" />
  <img src="./insight/img/plot_corr_with_target.png" width="100%" /> 
   <img src="./insight/img/plot_key_corr_and_impacts.png" width="100%" />
   <img src="./insight/img/plot_site_survival_year.png" width="100%" />
   <img src="./insight/img/plot_stage_surgery_gender_age.png" width="100%" />
  <img src="./insight/img/plot_yearly_event_and_classes.png" width="100%" /> 
   <img src="./insight/img/plot_survival_months.png" width="100%" />
    
   
### [ 타겟 레이블 상관관계 분석 ]  
  <img src="./insight/img/plot_target_extras.png" width="100%" />
    <img src="./insight/img/plot_target_label_by_surgery.png" width="100%" />
     <img src="./insight/img/plot_target_label_by_gender.png" width="100%" />
     <img src="./insight/img/figure.png" width="100%" />

###  [  모델 입력 특성 ] 

<table style='width:100%; border-collapse:collapse; font-size:0.95rem;'>
  <tr><td><b>Sex</b></td><td>성별</td></tr>
  <tr><td><b>Age recode with &lt;1 year olds and 90+</b></td><td>연령대</td></tr>
  <tr><td><b>Year of diagnosis</b></td><td>진단 연도</td></tr>
  <tr><td><b>Race recode (W, B, AI, API)</b></td><td>인종 재코드</td></tr>
  <tr><td><b>Primary Site</b></td><td>암 발생 부위</td></tr>
  <tr><td><b>Primary Site - labeled</b></td><td>암 발생 부위 라벨</td></tr>
  <tr><td><b>Derived Summary Grade 2018 (2018+)</b></td><td>암세포 악성도 등급</td></tr>
  <tr><td><b>Laterality</b></td><td>장기 좌우 구분</td></tr>
  <tr><td><b>EOD Schema ID Recode (2010+)</b></td><td>암 진행 단계 분류 코드</td></tr>
  <tr><td><b>Combined Summary Stage with Expanded Regional Codes (2004+)</b></td><td>암 진행 단계</td></tr>
  <tr><td><b>RX Summ--Surg Prim Site (1998+)</b></td><td>수술 코드</td></tr>
  <tr><td><b>RX Summ--Scope Reg LN Sur (2003+)</b></td><td>림프절 절제 범위</td></tr>
  <tr><td><b>RX Summ--Surg Oth Reg/Dis (2003+)</b></td><td>기타 수술</td></tr>
  <tr><td><b>Sequence number</b></td><td>암 확진 횟수</td></tr>
  <tr><td><b>Median household income inflation adj to 2023</b></td><td>가구 소득(2023 물가보정)</td></tr>
  <tr><td><b>EOD Primary Tumor Recode (2018+)</b></td><td>암의 전이 정도</td></tr>
  <tr><td><b>PRCDA 2020</b></td><td>PRCDA 2020</td></tr>
</table>

## 6. **인공지능 학습 결과서**
---

### 6-1. 기본 모델 아이디어
 해당 시스템은 환자에 대한 사망원인별 연속적인 사망률을 예측하여, 고위험 요인에 대한 선제적 조치와 예방을 목적으로 한다.  
 따라서 여러 시간대별 사망률을 예측할 수 있는 모델을 필요로 하고, 해당 형태를 구현하기 위하여 딥러닝 모델을 구현하여 사용하였다.  
 기본 모델의 형태와 손실함수는 *DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks.* (Lee, Changhee, et al., 2018)를 기반으로 작성하였다.  

 <div align="center">
    <img src="./data/images/original model.png" alt="Deephit original" style="max-width: 100%; height: auto; margin-bottom: 20px;"/>  
    <br>
    <i> Deephit 모델의 기본 구조 </i>
</div>


  DeepHit모델은 특성을 공유 Branch와 각 사건별 Branch에 차례대로 통과시켜 이산화시킨 시간 별 사건 발생 확률을 예측하는 형태의 모델이다.

### 6-2. 모델 개선 아이디어
  Deephit 모델은 시간대별 사건 발생 확률을 예측하는 MLP 기반의 모델로, 학습 과정에 여러 모듈을 추가하여 성능 개선을 도모할 수 있다.

#### 6-2.1. SEBlock (Squeeze-and-Excitation Block)
  StandardScaler, MinMaxScaler 대신 특성 중요도에 따라 특성마다 서로 다른 가중치를 줄 수 있게 하는 모듈을 삽입하여 성능을 높일 수 있다.    
  *Squeeze-and-Excitation Networks (Jie Hu, Li Shen, Gang Sun, 2018)* 에서 사용된 SEBlock 아이디어를 단순 특성 MLP에 적용하여 특성을 변환

#### 6-2.2. Residual Connection, Feature-wise Concat
  잔차 또는 기존 특성을 모델 중간에 삽입해주어 학습률을 높이고 성능을 안정화 시킬 수 있다.
   
  > 모델의 깊이가 깊지 않아 성능에 도움을 주지 않아 최종 모델에서는 사용하지 않음

#### 6-2.3. 1D, 2D CNN
  모델의 결과에 시간대별, 사건별 연관성을 추가하기 위하여 CNN을 사용

### **📊 모델별 학습 결과**

#### **모델 성능 평가 지표**
- Concordance Index (C-index) : 임의의 사건 두 개를 뽑아서 어떤 사건이 더 먼저 발생했는지 비교했을때, 해당 비교에 대한 정확도
- Integrated Brier Score (IBS) : 모델의 시간대별 오차 제곱의 평균 (MSE의 시간축에 대한 적분)
- MAE of predicted time : 모델이 예측한 생존 기간과, 실제 생존 기간 사이의 평균 오차

<table>
  <tr>
    <th>활용모델</th>
    <th>Concordance Index (C-index)</th>
    <th>Integrated Brier Score (IBS)</th>
    <th>MAE of predicted time</th>
    <th>Hyper parameters</th>
  </tr>
  <tr>
    <th scope="row" style="text-align:left;">SEBlock</th>
    <td>0.6412</td><td>0.2130</td><td>3.2790</td>
    <td>Hidden layer: (128, 64),<br> Time bins : 91,<br> SE ratio : 0.25</td>
  </tr>
  <tr>
    <th scope="row" style="text-align:left;">SEBlock + Feature concat</th>
    <td>0.5558</td><td>0.2463</td><td>4.6994</td>
    <td>Hidden layer: (128, 64),<br>  
      Time bins : 91,<br>  
      SE ratio : 0.25</td>
  </tr>
  <tr>
    <th scope="row" style="text-align:left;">SEBlock + 1-dimensional CNN</th>
    <td>0.7302</td><td>0.2116</td><td>3.6853</td>
    <td>Hidden layer: (128, 64),<br> Time bins : 91,<br> 
      SE ratio : 0.25,<br> CNN kernel size : 3</td>
  </tr>
  <tr>
    <th scope="row" style="text-align:left;">SEBlock + 2-dimensional CNN</th>
    <td>0.8263</td><td>0.2005</td><td>2.9128</td>
    <td>Hidden layer: (128, 64),<br> Time bins : 91,<br> SE ratio : 0.25,<br> CNN kernel size: (2,5), (2,3)</td>
  </tr>
</table>


### 6-3. 최종 모델 형태

<div align="center">
  <img src="./data/images/model2.png" alt="Deephit SEBlock" style="max-width: 100%; height: auto; margin-bottom: 20px;"/>
  <br>
  <i> 최종 Deephit 모델 </i>
</div>

<br>

<div align="center">
  <img src="./data/images/Deephit CNN.png" alt="Deephit SEBlock" style="max-width: 100%; height: auto; margin-bottom: 20px;"/>
  <br>
  <i> 최종 모델의 상세 구조 </i>
</div>

### 6-4. 위험 점수 계산

모델을 통해 얻은 시간대별 사건 발생 확률(pmf)에 대한 가중치합을 계산한 후 Sigmoid를 통하여 0~100점 사이의 점수로 변환    
가까운 시일 내에 사망할 확률이 높을 수록 높은 위험점수를 나타냄

### 6-5. 하이퍼파라미터 튜닝 및 학습 최적화
기본 파라미터로 학습하였을 때, 가장 좋은 성능을 보인 모델에 대해서 하이퍼 파라미터 튜닝과 학습 최적화를 진행
#### **학습 최적화 결과**
Batch Normalization + 스케줄러를 이용한 학습 최적화 결과

<table>
  <tr>
    <th>Concordance Index (C-index)</th>
    <th>Integrated Brier Score (IBS)</th>
    <th>MAE of predicted time</th>
  </tr>
  <tr>
    <td>0.9144</td><td>0.1638</td><td>5.3622</td>
  </tr>
</table>


## 7. **수행 결과**
---

---

## 8. **트러블 슈팅**

여러 파일을 인코딩하는 과정에서, 서로 다른 파일을 불러오면 같은 값이 서로 다르게 라벨링됨
-> 인코딩 전후 데이터를 매핑하여 딕셔너리 형태로 저장하고, 새로운 데이터를 불러올 때 딕셔너리 값을 통해 인코딩하고 새로운 값은 추가하여 업데이트 하는 방식으로 해결

모델 출력 결과에 softmax를 적용하면, 모든 확률값의 합이 1이 되어 환자가 기간 내에 무조건 사망한다 라는 결과가 나옴
-> 데이터의 마지막에 dummy 시간축을 추가하여, 관측기간 동안 환자가 사망하지 않을 확률도 고려

---

## 9. **한줄 회고**

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












