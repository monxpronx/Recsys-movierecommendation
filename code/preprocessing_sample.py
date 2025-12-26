import pandas as pd
import numpy as np

def main():
    sample_training_df = pd.read_csv("../data/train/sample_train_ratings.csv")
    sample_unique_users = sample_training_df["user"].unique()

    sample_ratio = 0.2
    n_sample_users = int(len(sample_unique_users) * sample_ratio)
    
    np.random.seed(42)
    sampled_users_id = np.random.choice(
        sample_unique_users, size=n_sample_users, replace=False
    )

    sampled_ratings = sample_training_df[sample_training_df['user'].isin(sampled_users_id)]
    valid_items = sampled_ratings['item'].unique()

    genres_df = pd.read_csv("../data/train/genres.tsv", sep="\t")

    genres_df = genres_df[genres_df['item'].isin(valid_items)]

    array, index = pd.factorize(genres_df["genre"])
    genres_df["genre"] = array
    genres_df.groupby("item")["genre"].apply(list).to_json(
        "../data/Ml_sample_item2attributes.json"
    )


if __name__ == "__main__":
    main()
