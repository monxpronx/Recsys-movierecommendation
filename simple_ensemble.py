import pandas as pd

# 1. 파일 로드 (경로 맞춰주세요)
sasrec = pd.read_csv("./submission_SASRec_top100.csv") # score 컬럼 필수
lightgcn = pd.read_csv("./submission_LightGCN_top100.csv")
ease = pd.read_csv("./submission_EASE_top100.csv")

# 2. 점수 정규화 (Min-Max Scaling) - 모델마다 점수 스케일이 다르므로 필수!
def normalize(df):
    df['score'] = (df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min())
    return df

sasrec = normalize(sasrec)
lightgcn = normalize(lightgcn)
ease = normalize(ease)

# 3. 데이터 합치기 (User, Item 기준으로)
# merge를 여러번 해서 점수를 옆으로 붙입니다.
merged = pd.merge(ease, sasrec, on=['user', 'item'], how='outer', suffixes=('_ease', '_sas'))
merged = pd.merge(merged, lightgcn, on=['user', 'item'], how='outer')
merged.rename(columns={'score': 'score_light'}, inplace=True)

# 결측치(한 모델에는 있는데 다른 모델엔 없는 경우) 0으로 채움
merged.fillna(0, inplace=True)

# 4. 가중치 적용 (Weighted Sum) - 여기가 핵심!
# EASE가 1등이니 가장 높은 가중치 부여
merged['final_score'] = (merged['score_ease'] * 0.7) + \
                        (merged['score_sas'] * 0.15) + \
                        (merged['score_light'] * 0.15)

# 5. 최종 Top-10 뽑기
merged.sort_values(['user', 'final_score'], ascending=[True, False], inplace=True)
submission = merged.groupby('user').head(10)[['user', 'item']]

submission.to_csv("weighted_ensemble_submission3.csv", index=False)
print("완료! weighted_ensemble_submission.csv 생성됨")