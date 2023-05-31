![header](https://github.com/boostcampaitech5/level2_dkt-recsys-09/assets/91173904/3265b386-e149-4ebc-ae98-cff6d00c6281)

RecSys 09조 FFM   

## 팀원 소개

![그림2](https://github.com/boostcampaitech5/level2_dkt-recsys-09/assets/91173904/c084f436-be9b-42cb-8aa8-02bdd4db7e9a)

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
- [5. 결과](#4-결과)
### [결론 및 개선 방안](#결론-및-개선-방안-1)

## Project Configuration
📦level2_dkt-recsys-09  
 ┣ 📂.github  
 ┃ ┣ 📂ISSUE_TEMPLATE  
 ┃ ┃ ┣ 📜기능-수정.md  
 ┃ ┃ ┣ 📜버그-발견.md  
 ┃ ┃ ┗ 📜새로운-기능-추가.md  
 ┃ ┗ 📜PULL_REQUEST_TEMPLATE.md  
 ┣ 📂DKT  
 ┃ ┣ 📂base  
 ┃ ┃ ┣ 📜base_data_loader.py  
 ┃ ┃ ┣ 📜base_model.py  
 ┃ ┃ ┣ 📜base_trainer.py  
 ┃ ┃ ┗ 📜__init__.py  
 ┃ ┣ 📂config  
 ┃ ┃ ┣ 📜config_HM.json  
 ┃ ┃ ┣ 📜config_LGBM.json  
 ┃ ┃ ┣ 📜config_lgcntrans.json  
 ┃ ┃ ┗ 📜config_ultraGCN.json  
 ┃ ┣ 📂data_loader  
 ┃ ┃ ┣ 📜dataloader_lgcnlstmattn.py  
 ┃ ┃ ┣ 📜dataloader_practice.py  
 ┃ ┃ ┣ 📜data_loaders_GCN.py  
 ┃ ┃ ┣ 📜data_preprocess_GCN.py  
 ┃ ┃ ┣ 📜data_preprocess_HM.py  
 ┃ ┃ ┣ 📜data_preprocess_LQ.py  
 ┃ ┃ ┣ 📜feature_engine.py  
 ┃ ┃ ┣ 📜make_user_item_interaction.py  
 ┃ ┃ ┣ 📜preprocess_lgcntrans.py  
 ┃ ┃ ┣ 📜preprocess_ML.py  
 ┃ ┃ ┗ 📜__init__.py  
 ┃ ┣ 📂logger  
 ┃ ┃ ┣ 📜logger.py  
 ┃ ┃ ┣ 📜logger_config.json  
 ┃ ┃ ┣ 📜visualization.py  
 ┃ ┃ ┗ 📜__init__.py  
 ┃ ┣ 📂model  
 ┃ ┃ ┣ 📜loss_GCN.py  
 ┃ ┃ ┣ 📜metric_GCN.py  
 ┃ ┃ ┣ 📜model_GCN.py  
 ┃ ┃ ┣ 📜model_lgcnlstmattn.py  
 ┃ ┃ ┣ 📜model_LQ.py  
 ┃ ┃ ┣ 📜model_ML.py  
 ┃ ┃ ┗ 📜__init__.py  
 ┃ ┣ 📂src  
 ┃ ┃ ┣ 📜criterion.py  
 ┃ ┃ ┣ 📜feature_engine.py  
 ┃ ┃ ┣ 📜metric.py  
 ┃ ┃ ┣ 📜optimizer.py  
 ┃ ┃ ┣ 📜scheduler.py  
 ┃ ┃ ┣ 📜utils.py  
 ┃ ┃ ┗ 📜__init__.py  
 ┃ ┣ 📂trainer  
 ┃ ┃ ┣ 📜trainer_GCN.py  
 ┃ ┃ ┣ 📜trainer_HM.py  
 ┃ ┃ ┣ 📜trainer_lgcnlstmattn.py  
 ┃ ┃ ┣ 📜trainer_LQ.py  
 ┃ ┃ ┣ 📜trainer_ML.py  
 ┃ ┃ ┗ 📜__init__.py  
 ┃ ┣ 📂utils  
 ┃ ┃ ┣ 📜util.py  
 ┃ ┃ ┗ 📜__init__.py  
 ┃ ┣ 📜.gitignore  
 ┃ ┣ 📜args_LQ.py  
 ┃ ┣ 📜parse_config.py  
 ┃ ┣ 📜requirements.txt  
 ┃ ┣ 📜test_GCN.py  
 ┃ ┣ 📜test_HM.py  
 ┃ ┣ 📜test_lgcnlstmattn.py  
 ┃ ┣ 📜test_LQ.py  
 ┃ ┣ 📜test_ML.py  
 ┃ ┣ 📜train_GCN.py  
 ┃ ┣ 📜train_lgcnlstmattn.py  
 ┃ ┣ 📜train_LQ.py  
 ┃ ┗ 📜train_ML.py  
 ┣ 📂eda  
 ┃ ┣ 📜eda_integration.ipynb  
 ┃ ┣ 📜hcw_eda.ipynb  
 ┃ ┣ 📜khj_eda.ipynb  
 ┃ ┣ 📜khw_eda.ipynb  
 ┃ ┣ 📜lhm_eda.ipynb  
 ┃ ┗ 📜mhj_eda.ipynb  
 ┣ 📂expriments  
 ┃ ┣ 📂bert4rec  
 ┃ ┃ ┗ 📜bert4rec.ipynb  
 ┃ ┣ 📂ultragcn_feature  
 ┃ ┃ ┣ 📜config_ultraGCN_feature.json  
 ┃ ┃ ┣ 📜data_preprocess_GCN.py  
 ┃ ┃ ┗ 📜model_GCN.py  
 ┃ ┣ 📂UltraGCN_ii_matrix  
 ┃ ┃ ┣ 📜data_preprocess_GCN.py  
 ┃ ┃ ┣ 📜loss_GCN.py  
 ┃ ┃ ┗ 📜model_GCN.py  
 ┃ ┣ 📜#6_FeatureEngineering.ipynb  
 ┃ ┣ 📜cv_baseline.ipynb  
 ┃ ┗ 📜LGBM_baseline.ipynb  
 ┗ 📜README.md  
 
 
