import numpy as np
import pandas as pd
from functools import reduce
from scipy.special import erfinv

# 1. 파일 로드 (경로 맞춰주세요)
sasrec = pd.read_csv("./submission_SASRec_top100.csv") # score 컬럼 필수
s3rec = pd.read_csv("./submission_S3Rec_top100.csv")
gru4rec = pd.read_csv("./submission_GRU4Rec_top100.csv")
gru4recF = pd.read_csv("./submission_GRU4RecF_top100.csv")
ease = pd.read_csv("./submission_EASE_top100.csv")
admm = pd.read_csv("./submission_ADMM_top100.csv")
cdae = pd.read_csv("./submission_CDAE_top100.csv")
multivae = pd.read_csv("./submission_MultiVAE_top100.csv")
recvae = pd.read_csv("./submission_RecVAE_top100.csv")
lightgcn = pd.read_csv("./submission_LightGCN_top100.csv")

# 2. 점수 정규화 (Min-Max Scaling) - 모델마다 점수 스케일이 다르므로 필수!
def normalize(df):
    df['score'] = (df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min())
    return df

sasrec = normalize(sasrec)
s3rec = normalize(s3rec)
gru4rec = normalize(gru4rec)
gru4recF = normalize(gru4recF)
ease = normalize(ease)
admm = normalize(admm)
cdae = normalize(cdae)
multivae = normalize(multivae)
recvae = normalize(recvae)
lightgcn = normalize(lightgcn)

# 3. 데이터 합치기 (User, Item 기준으로)
# 모델 dict로 관리
models = {
    #'sasrec': sasrec,
    #'s3rec': s3rec,
    #'gru4rec': gru4rec,
    'gru4recF': gru4recF,
    #'ease': ease,
    'admm': admm,
    #'cdae': cdae,
    #'multi': multivae,
    #'recvae': recvae,
    'light': lightgcn
}

# score 컬럼 이름 통일
dfs = []
for name, df in models.items():
    tmp = df[['user', 'item', 'score']].copy()
    tmp.rename(columns={'score': f'score_{name}'}, inplace=True)
    dfs.append(tmp)

# reduce로 outer merge
merged = reduce(
    lambda left, right: pd.merge(
        left, right,
        on=['user', 'item'],
        how='outer'
    ),
    dfs
)

# 결측치(한 모델에는 있는데 다른 모델엔 없는 경우) 0으로 채움
merged.fillna(0, inplace=True)

# 4. 가중치 적용 (Weighted Sum) - 여기가 핵심!
merged['final_score'] = (merged['score_admm'] * 0.7 +
                        #merged['score_ease'] * 0.4 + 
                        #merged['score_sasrec'] * 0.2 +
                        #merged['score_s3rec'] * 0.2
                        merged['score_gru4recF'] * 0.2 +
                        merged['score_light'] * 0.1
                        #merged['score_cdae'] * 0.1
                        #(merged['score_multi'] * 0.1) + \
                        #(merged['score_recvae'] * 0.1)
                        )

# 5. 최종 Top-10 뽑기
merged.sort_values(['user', 
                    'final_score'
                    ], ascending=[True, False], inplace=True)
submission = merged.groupby('user').head(10)[['user', 'item']]

submission.to_csv("weighted_ensemble_submission.csv", index=False)
print("완료! weighted_ensemble_submission.csv 생성됨")