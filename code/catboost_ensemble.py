# pip install catboost

import argparse
import os
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from sklearn.preprocessing import MinMaxScaler

def parse_args():
    parser = argparse.ArgumentParser(description="Train CatBoost & Rerank Candidates")
    
    # 입력: 3개 모델의 결과 CSV (user, item, score)
    parser.add_argument("--sasrec", type=str, required=False, help="Path to SASRec result csv")
    parser.add_argument("--lightgcn", type=str, required=False, help="Path to LightGCN result csv")
    parser.add_argument("--ease", type=str, required=False, help="Path to EASE result csv")

    # 입력: 전처리된 메타데이터 경로(preprocess.py 결과물)
    parser.add_argument("--processed_dir", type=str, default="../data/train", help="Path to the preprocessed files")
    
    # 출력
    parser.add_argument("--output_file", type=str, default="../data/eval", help="Path to save the final submission csv")

    return parser.parse_args()

def load_and_normalize(path, col_name):
    # CSV를 읽고 score 컬럼을 0~1 사이로 정규화합니다.
    print(f"Loading {col_name} from {path}...")
    if not os.path.exists(path):
        print(f"[Error] File not found: {path}")
        return None
    
    df = pd.read_csv(path)
    # column명 변경 (ex. score -> sasrec_score)
    df.rename(columns={"score": col_name}, inplace=True)

    # 정규화 (Min-Max Scaling)
    scaler = MinMaxScaler()
    df[col_name] = scaler.fit_transform(df[[col_name]])
    return df

def main():
    args = parse_args()

    # --------------------------------------------------------------------------
    # 1. Load Model Results (Candidate Generation)
    # --------------------------------------------------------------------------
    ########## 모델 추가할거면 변경 ##########
    if args.sasrec:
        df_sas = load_and_normalize(args.sasrec, "sasrec_score")
    if args.lightgcn:
        df_lightgcn = load_and_normalize(args.lightgcn, "lightgcn_score")
    if args.ease:
        df_ease = load_and_normalize(args.ease, "ease_score")
    
    # Outer Join으로 병합
    print("Merging candidates from 3 models...")
    merged_df = df_sas
    merged_df = pd.merge(merged_df, df_lightgcn, on=["user", "item"], how="outer")
    merged_df = pd.merge(merged_df, df_ease, on=["user", "item"], how="outer")
    # merged_df: (sasrec_score, lightgcn_score, ease_score)

    # 추천 안 된 모델의 점수는 0점 처리
    merged_df.fillna(0, inplace=True)  # 결측치는 0으로 채움

    print(f"Merged model results: {len(merged_df)}")

    # --------------------------------------------------------------------------
    # 2. Load Preprocessed Metadata
    # --------------------------------------------------------------------------
    print("Loading preprocessed metadata and training data...")
    # item_meta: (item, release_age, director, writer, genre)
    item_meta = pd.read_csv(os.path.join(args.processed_dir, "item_metadata.csv"))
    # user_feat: (user, view_count)
    user_feat = pd.read_csv(os.path.join(args.processed_dir, "user_features.csv"))

    train_path = os.path.join(args.processed_dir, "catboost_train_set.csv")
    if not os.path.exists(train_path):
        print(f"[Error] catboost train data not found: {train_path}")
        return
    # train_df: (user, item, target)
    train_df = pd.read_csv(train_path)

    # --------------------------------------------------------------------------
    # 3. CatBoost 학습 (Metadata & User History 학습))
    # --------------------------------------------------------------------------
    # Train 데이터에 메타데이터 붙이기
    train_df = train_df.merge(item_meta, on="item", how="left")
    train_df = train_df.merge(user_feat, on="user", how="left")

    # 학습에 사용할 Feature (모델 Score는 학습 데이터에 없으므로 제외하고 metadata만 사용)
    # train_df:(user, item, target, release_age, director, writer, genre, view_count)
    exclude_cols = ["user", "item", "target"]
    # feature_cols: ['release_age', 'director', 'writer', 'genre', 'view_count']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    cat_features = ["director", "writer", "genre"] # 범주형 변수
    num_features = ["release_age", "view_count"] # 수치형 변수

    for c in cat_features:
        train_df[c] = train_df[c].astype(str).fillna('Unknown')
    for c in num_features:
        train_df[c] = train_df[c].fillna(-1)

    print(f"Training Features: {feature_cols}")
    print("Training CatBoostRanker...")

    train_df.sort_values(by='user', inplace=True, kind='mergesort')  # Stable sort for group_id
    train_df.reset_index(drop=True, inplace=True)

    train_pool = Pool(
        data=train_df[feature_cols],
        label=train_df["target"],
        cat_features=cat_features,
        group_id=train_df["user"]
    )

    model = CatBoostRanker(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        loss_function='YetiRank',
        eval_metric='RecallAt:top=10',
        verbose=100,
        random_seed=42,
        task_type='GPU'
    )

    model.fit(train_pool)

    # --------------------------------------------------------------------------
    # 4. 최종 예측 및 앙상블 (Inference)
    # --------------------------------------------------------------------------
    print("Preparing CatBoost scores for candidates...")
          
    # 후보군(merged_df)에 metadata 붙이기
    test_df = merged_df.merge(item_meta, on="item", how="left")
    test_df = test_df.merge(user_feat, on="user", how="left")
    # test_df:(user, item, sasrec_score, lightgcn_score, ease_score, release_age, director, writer, genre, view_count)
    for c in cat_features:
        test_df[c] = test_df[c].astype(str).fillna('Unknown')
    for c in num_features:
        test_df[c] = test_df[c].fillna(-1)

    # CatBoost 예측
    test_df['catboost_score'] = model.predict(test_df[feature_cols])
    # feature_cols: ['release_age', 'director', 'writer', 'genre', 'view_count']

    # 점수 정규화 및 가중 합산 (Ensemble)
    scaler = MinMaxScaler()
    test_df['catboost_score'] = scaler.fit_transform(test_df[['catboost_score']])

    print("Calculating Final Weighted Scores...")
    # 가중치 설정 (실험적으로 조절 가능)
    w_cat = 0.3
    w_sas = 0.1
    w_light = 0.1
    w_ease = 0.5

    test_df['final_score'] = (
        w_cat * test_df['catboost_score'] +
        w_sas * test_df['sasrec_score'] +
        w_light * test_df['lightgcn_score'] +
        w_ease * test_df['ease_score']
    )

    # --------------------------------------------------------------------------
    # 5. Top-10 선정 및 저장
    # --------------------------------------------------------------------------
    print("Selecting Top-10 items per user...")
    test_df.sort_values(by=['user', 'final_score'], ascending=[True, False], inplace=True)

    test_df.drop_duplicates(subset=['user', 'item'], keep='first', inplace=True)

    final_submission = test_df.groupby('user').head(10)[['user', 'item']]

    final_submission.to_csv(args.output_file + "_catboost_reranked.csv", index=False)
    print(f"[Done] Successfully saved to {args.output_file}_catboost_reranked.csv")
    print("\nFeature Importance:")
    try:
        print(dict(zip(feature_cols, model.get_feature_importance())))
    except:
        pass

if __name__ == "__main__":
    main()