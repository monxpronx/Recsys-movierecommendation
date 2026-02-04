# Movie Recommendation (RecSys)

본 프로젝트는 Upstage AI Stages에서 주최한 추천 시스템 대회 기반으로 수행하였습니다.  
사용자의 영화 시청 이력 데이터를 바탕으로 사용자가 다음에 시청할 영화 및 선호할 영화를 예측하는 추천 시스템입니다.


## Problem Overview
- 추천 시스템(RecSys) 문제
- 사용자 시청 이력(timestamp 기반)을 활용한 영화 추천
- Implicit Feedback을 중심으로 한 Sequential Recommendation 문제
- 일부 item이 누락(dropout)된 sequence를 기반으로 추천 수행


## Evaluation
- 평가 지표: Recall@K
- 사용자별 Top-K 추천 결과에 대해 정답 아이템 포함 여부로 평가
- Recall 값이 높을수록 모델 성능이 우수


## Dataset
본 데이터는 대회에서 제공된 공개 데이터셋을 사용하였습니다.

- 사용자–아이템 시청 이력 데이터 (`train_ratings.csv`)
- 영화 메타데이터 (`제목, 장르, 개봉연도 등 각 .tsv`)
- 전처리를 통해 생성된 아이템 속성 데이터 (`ML_item2attributes.json`)
