import argparse
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='saved')
    parser.add_argument('--output_path', type=str, default='submissions')

    args = parser.parse_args()

    rec_file = f"{args.save_path}/{args.model}/{args.dataset}/recommendation/{args.model}-{args.dataset}-top{args.topk}.csv"
    df = pd.read_csv(rec_file)

    df = df.sort_values(['user_id', 'score'], ascending=[True, False])
    submission = df[['user_id', 'item_id']]
    submission.columns = ['user', 'item']

    os.makedirs(args.output_path, exist_ok=True)
    out_path = f"{args.output_path}/{args.model.lower()}_top{args.topk}.csv"
    submission.to_csv(out_path, index=False)

    print(f"submission 생성 완료 → {out_path}")

if __name__ == "__main__":
    main()
