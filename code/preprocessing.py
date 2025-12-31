import argparse
import os
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Static Data & User Features")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/train",
        help="Directory containing the raw data files",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../data/train",
        help="Directory to save the preprocessed data files",
    )
    return parser.parse_args()

def prune_categoricals(df, col_name, threshold=5):
    value_counts = df[col_name].value_counts()
    to_prune = set(value_counts[value_counts < threshold].index)
    df[col_name] = df[col_name].where(df[col_name].isin(to_prune), "Other")
    print(f"[{col_name}] Pruned: {len(value_counts)} -> {len(to_prune)} +1 'Other'")
    return df

def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    print(f"Loading raw data from {args.data_dir}...")

    # --------------------------------------------------------------------------
    # 1. Load Raw Data
    # --------------------------------------------------------------------------
    try:
        years = pd.read_csv(os.path.join(args.data_dir, "years.tsv"), sep="\t")
        genres = pd.read_csv(os.path.join(args.data_dir, "genres.tsv"), sep="\t")
        directors = pd.read_csv(os.path.join(args.data_dir, "directors.tsv"), sep="\t")
        writers = pd.read_csv(os.path.join(args.data_dir, "writers.tsv"), sep="\t")
        train_ratings = pd.read_csv(os.path.join(args.data_dir, "train_ratings.csv"), sep=",")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # --------------------------------------------------------------------------
    # 2. Process Static Metadata (Item 기준)
    # --------------------------------------------------------------------------
    print("Processing Item Metadata...")

    # (1) Years -> Release Age
    CURRENT_YEAR = 2025
    # years.columns = ["item", "year"]
    years["release_age"] = CURRENT_YEAR - years["year"]
    # years['year'] = years['year'].fillna(2000).astype(int) # 어차피 non-null

    # (2) Directors & Writers -> Pruning
    # directors.columns = ["item", "director"]
    # writers.columns = ["item", "writer"]
    directors = prune_categoricals(directors, "director", threshold=5)
    writers = prune_categoricals(writers, "writer", threshold=5)

    # 같은 item이면 먼저 등장한 director, writer만 대표로 남김
    directors = directors.drop_duplicates('item')
    writers = writers.drop_duplicates('item')

    # (3) Genres
    # genres.columns = ["item", "genre"]
    genres = genres.drop_duplicates('item') # 같은 item이면 먼저 등장한 genre 하나만 남김

    # Merge All Item Features
    item_meta = years[['item', 'release_age']]
    item_meta = item_meta.merge(directors, on="item", how="left")
    item_meta = item_meta.merge(writers, on="item", how="left")
    item_meta = item_meta.merge(genres, on="item", how="left")

    # item_meta.fillna('unknown', inplace=True) 어차피 non-null

    # 저장
    item_save_path = os.path.join(args.output_path, "item_metadata.csv")
    item_meta.to_csv(item_save_path, index=False)
    print(f"Saved processed item metadata to {item_save_path}")

    # --------------------------------------------------------------------------
    # 3. Process User Features (User 기준))
    # --------------------------------------------------------------------------
    print("Processing User Features...")

    # 유저 활동량 (Activitiy Count)
    user_activity = train_ratings.groupby("user")["item"].count().reset_index()
    user_activity.columns = ["user", "view_count"]

    # 저장
    user_save_path = os.path.join(args.output_path, "user_features.csv")
    user_activity.to_csv(user_save_path, index=False)
    print(f"Saved processed user features to {user_save_path}")

    # --------------------------------------------------------------------------
    # 4. Create Train Data for CatBoost (Validation Split)
    # --------------------------------------------------------------------------
    print("Creating Training Set for CatBoost (Leave-One-Out)...")

    # (1) 유저가 지금까지 본 모든 영화 리스트를 집합(set)으로 만듦
    # user_history: {user_id: {item1, item2...}}
    user_history = train_ratings.groupby("user")["item"].apply(set).to_dict()

    # (2) 전체 아이템 리스트
    all_items = train_ratings["item"].unique()

    # (3) 마지막 아이템(정답) 준비
    # 타임스탬프 기준 정렬 (이미 정렬))
    if 'timestamp' in train_ratings.columns:
        train_ratings = train_ratings.sort_values(by=["user", "timestamp"])

    last_items = train_ratings.groupby('user').tail(1)
    last_items['target'] = 1  # 정답 레이블

    users = last_items['user'].values

    # (4) Negative Sampling (본 거 뺴고 뽑기)
    neg_rows = []
    
    # 랜덤 시드 고정
    np.random.seed(42)

    print("Generating Negative Samples...")
    for u in users:
        seen_items = user_history.get(u, set())
        
        # 4개를 뽑을 때까지 반복
        count = 0
        while count < 4:
            # 랜덤하게 아이템 하나 선택
            neg_item = np.random.choice(all_items)

            # 본 적 없는 아이템이면 추가
            if neg_item not in seen_items:
                neg_rows.append({'user': u, 'item': neg_item, 'target': 0})
                count += 1
    
    neg_df = pd.DataFrame(neg_rows)

    # (5) Positive + Negative 합치기
    train_set = pd.concat([last_items[['user', 'item', 'target']], neg_df])
                                       
    # 저장
    train_save_path = os.path.join(args.output_path, "catboost_train_set.csv")
    train_set.to_csv(train_save_path, index=False)
    print(f"Saved processed training data to {train_save_path}")

if __name__ == "__main__":
    main()
