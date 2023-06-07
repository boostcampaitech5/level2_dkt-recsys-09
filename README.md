![그림1](https://github.com/boostcampaitech5/level2_dkt-recsys-09/assets/91173904/44551850-8a5a-445f-895d-7860600d87d2)

RecSys 09조 FFM   

## 팀원 소개

![그림2](https://github.com/boostcampaitech5/level2_dkt-recsys-09/assets/91173904/d35b51d9-d093-4c23-9c2b-40674362eba1)

## 목차
### [Project Configuration](#project-configuration-1)
### [프로젝트 개요](#프로젝트-개요-1)
- [1. 프로젝트 주제 및 목표](#1-프로젝트-주제-및-목표)
- [2. 프로젝트 개발 환경](#2-프로젝트-개발-환경)
### [프로젝트 팀 구성 및 역할](#프로젝트-팀-구성-및-역할-1)
### [프로젝트 수행 내용 및 결과](#프로젝트-수행-내용-및-결과-1)
- [1. EDA](#1-eda)
- [2. Feature Engineering](#2-feature-engineering)
- [3. 모델링](#3-모델링)
- [4. 성능 개선 및 앙상블](#4-성능-개선-및-앙상블)
- [5. 결과](#5-결과)
### [결론 및 개선 방안](#결론-및-개선-방안-1)

## Project Configuration
📦level2_dkt-recsys-09  
 ┣ 📂DKT  
 ┃ ┣ 📂base  
 ┃ ┣ 📂config  
 ┃ ┣ 📂data_loader  
 ┃ ┣ 📂logger    
 ┃ ┣ 📂model  
 ┃ ┣ 📂train  
 ┃ ┣ 📂test  
 ┃ ┣ 📂trainer  
 ┃ ┣ 📂utils  
 ┃ ┣ 📜.gitignore  
 ┃ ┣ 📜args_LQ.py  
 ┃ ┣ 📜parse_config.py  
 ┃ ┣ 📜requirements.txt  
 ┃ ┣ 📜select_test_model.py  
 ┃ ┣ 📜select_train_model.py  
 ┣ 📂eda  
 ┣ 📂experiments  
 ┗ 📜README.md  
 
## 프로젝트 개요

### 1. 프로젝트 주제 및 목표

Deep Knowledge Tracing (DKT)는 딥러닝을 활용해 지식 상태를 추적하는 방법론으로 개인에게 맞춤화 된 교육을 제공하는데 활용된다. 본 프로젝트에서는 Iscream데이터셋에서 주어진 학생들의 문제 풀이 내역을 기반으로 모델을 학습하여 마지막 문제의 정답 여부를 예측하는 것을 목표로 한다.

### 2. 프로젝트 개발 환경
•	팀 구성: 5인 1팀, 인당 V100 서버를 VS Code와 SSH로 연결하여 사용  
•	협업 환경: Notion, GitHub, Wandb


## 프로젝트 팀 구성 및 역할
![화면 캡처](https://github.com/boostcampaitech5/level2_dkt-recsys-09/assets/91173904/76eb918c-c070-41d2-9479-098655de0f9e)


## 프로젝트 수행 내용 및 결과

### 1. EDA

(1) UserID: 총 7,442명의 고유 사용자 번호  
(2) assessmentItemID: 문항의 고유번호  
(3) testId: 시험지의 고유번호  
(4) answerCode: 사용자가 푼 문항의 정답 여부를 담고 있는 이진 데이터  
(5) Timestamp: 사용자가 문항을 풀기 시작한 시점의 데이터  
(6) KnowledgeTag: 사용자가 푼 문항의 고유 태그 데이터  

### 2. Feature Engineering

- Timestamp기반으로 문제별 풀이시간 변수 생성
- Timestamp에서 시간 요일 추출
- 각 feature별로 정답여부 확인 (answerCode)
- 위의 feature를 정답과 오답을 기준으로 나눠서 생각
- testId의 1~3번째 index 값은 testcode라는 대분류로 보고 변수 생성
- testcode을 더 크게 범주화하는 testcode_cat 변수 생성
- Sequence Model을 사용하지 않고 일반적인 지도 학습 모델에서 사용하기 위해 주어진 데이터 이전/이후 데이터들을 포함하는 메모리를 feature로 포함시킴

### 3. 모델링

(1) UltraGCN
- GCN(Graph Convolutional Network) 기반의 모델로 user-item graph를 사용해 user와 item 사이의 link를 예측하는 모델
- UltraGCN은 layer가 무한히 많아질 때 수렴상태에 도달한 embedding을 학습  
  코사인 유사도를 최대화하는 loss function을 이용해 학습  
  loss function은 가중치를 가지는 BCE loss의 형태
- 다른 GCN 모델의 경우에는 oversmoothing을 해결하기 위해서 layer 수를 조정하지만, UltraGCN의 경우 layer 수가 무한하다고 가정하기 때문에 이를 negative sampling을 통해서 해결  
  DKT 데이터의 경우 정답(1)일 때의 정보뿐만 아니라 오답(0)일 때의 정보도 존재하므로 negative sampling을 하지 않

(2) LightGBM 
- 경사부스팅 기법을 사용하여 앙상블 학습을 수행. 약한 학습기들을 순차적으로 학습시켜 전체 모델을 구성하는 강력한 학습기를 만드는 방법
- Leaf-wise Tree Growth 방식을 사용하기 때문에, 균형 잡힌 트리구조를 생성하여 불필요한 분기를 줄이고 모델의 학습 속도를 향상
- 압축된 데이터 표현방식과 효율적인 메모리 관리 기법을 사용하여 메모리 사용량을 최소화. 이는 대규모 데이터셋에서도 상대적으로 적은 메모리를 요구하며, OOM 문제 해결을 도움
- 일반적으로 범주형 변수는 사전에 원핫인코딩 등의 전처리 과정이 필요하지만, LGBM은 범주형 변수를 정수형으로 인코딩하여 직접 사용. 전처리 과정을 단순화하고 메모리 사용량을 줄이는 것을 도움

(3) Hybrid Model (LightGCN + LSTMattn)
- Sequenetial한 모델을 위해서 Item만의 그래프로 나타내어야 했고 이를 위해 (전체 Item x 전체 Item) 크기의 라플라시안 행렬로 나타냄
- Graph Convolution을 통해 Item-Item Graph를 Embedding으로 나타냄

(4) LSTM + Last Query
- 다수의 Feature를 사용하지 않고 마지막 Query만 사용하여 시간 복잡도를 낮춤
- 문제 간 특징을 Transformer로 파악하고 일련의 Sequence 사이 특징들을 LSTM으로 뽑아낸 뒤에 마지막 DNN을 통해 Sequence 별 정답을 예측

(5) BERT4Rec
- 데이터 전처리 과정에서 userid와 testid에 따른 groupby를 진행하여 한 유저가 하나의 시험지를 푼 세션을 하나의 시퀀스로 정함. 한 유저가 동일한 시험지를 여러번 푼 경우, 전처리를 통해 마지막 경우의 데이터만 사용
- 시험지 별 최대 문항 개수인 13을 max sequence length로 잡고, 이 길이에 맞춰서 padding을 진행
- 모델의 Embedding layer는 인풋 시퀀스의 일반적인 임베딩을 맡는 token embedding과 sin, cos을 이용해 시퀀스의 위치 정보를 나타내는 positional embedding으로 구성
- transformer layer는 multihead attention layer와 pointwise feedforward layer로 구성되어 있으며 활성화 함수로 Gelu를 사용. 논문에서와 마찬가지로 본 구현에서는 2개의 head와 2개의 layer block을 사용

(6) Hybrid Model (UltraGCN + LSTM, LSTMattn)
- UltraGCN으로 학습한 item embedding을 sequential model의 item embedding으로 사용하여 모델을 학습

(7) Catboost
- 일반적으로 범주형 변수는 추가적인 전처리가 필요하지만 범주형 변수를 모델에 입력만 해주면 처리를 모델에서 직접 해줌

![그림3](https://github.com/boostcampaitech5/level2_dkt-recsys-09/assets/91173904/508fe7a6-7b11-4c10-bd9b-64706c7dbc38)

### 4. 성능 개선 및 앙상블

![그림6](https://github.com/boostcampaitech5/level2_dkt-recsys-09/assets/91173904/1c8ae40c-da91-4a88-88c4-353019aa69e7)

### 5. 결과

UltraGCN+LGBM+Catboost+Lgcntrans과 UltraGCN+LGBM+Lgcntrans+lastquery 앙상블 모델을 채택

## 결론 및 개선 방안

- 잘했던 점  
  다양한 협업 툴(GitHub, Wandb)을 사용해서 프로젝트를 진행했다.  
  특히 GitHub에서 issue 관리나 pull request 요청 등의 기능을 적극적으로 사용했다.  
  다양한 모델들, 특히 딥러닝 모델들을 만들어보려고 노력했다.  
- 시도했으나 잘 되지 않았던 점  
  높은 성능을 보여준 UltraGCN 모델을 기반으로 성능 개선을 위한 다양한 시도들을 해봤지만 큰 성과가 없었다.  
  Kaggle Riiid 대회 1등 모델인 Last Query 모델을 사용하여 Sequence로 인한 Transformer 시간 복잡도를 줄이기 위해 노력하였다.   하지만 이번 DKT 대회에서는 Sequence 길이에 의한 시간 복잡도가 문제가 되지 않았기에 기대한 만큼의 성능이 나오지 않았다.  
- 아쉬웠던 점  
  EDA에 기울인 노력에 비해서 feature engineering이나 feature selection에 노력을 덜 기울인 것 같아서 아쉽다.   
  DL 모델에 집중하느라 ML 모델은 많은 실험을 해보지 못한 것 같아서 아쉽다.  
  pytorch template을 사용해서 모델들을 모듈화하려 했지만 부족한 점이 있는 것 같아서 아쉽다.  
- 프로젝트를 통해 배운 점  
	 이번 대회에서 Private과 public의 결과가 매우 상이하게 나타났는데, 강건한 모델의 중요성을 느꼈다.  
	 시계열 데이터를 다뤄보았고, 시계열 데이터의 유의미함과 중요성을 느꼈다.  
	 GitHub를 통한 제대로 된 협업을 경험하면서 커뮤니케이션과 협업력의 중요성을 느꼈다.  

