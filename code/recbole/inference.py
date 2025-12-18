import argparse
import os
import pandas as pd

from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk

TOPK = 10
OUTPUT_DIR = "output"


def make_output_path(checkpoint_path: str) -> str:
    base = os.path.basename(checkpoint_path)
    name, _ = os.path.splitext(base)
    return os.path.join(OUTPUT_DIR, f"submission_{name}.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        args.checkpoint
    )

    uid_field = dataset.uid_field
    iid_field = dataset.iid_field

    test_uids = test_data.dataset.inter_feat[uid_field].unique().tolist()
    test_uids = sorted(test_uids)

    model.eval()

    topk_score, topk_iid = full_sort_topk(
        test_uids,
        model,
        test_data,
        k=TOPK,
        device=config["device"],
    )

    user_tokens = dataset.id2token(uid_field, test_uids)

    rows = []
    for idx, u in enumerate(user_tokens):
        item_inner_ids = topk_iid[idx].cpu().numpy().tolist()
        item_tokens = dataset.id2token(iid_field, item_inner_ids)

        for it in item_tokens:
            rows.append((u, it))

    out_df = pd.DataFrame(rows, columns=["user", "item"])

    for col in ["user", "item"]:
        try:
            out_df[col] = out_df[col].astype(int)
        except Exception:
            pass

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = make_output_path(args.checkpoint)
    out_df.to_csv(out_path, index=False)

    print(f"[OK] saved → {out_path}")
    print(f"users={len(test_uids)} | rows={len(out_df)}")


if __name__ == "__main__":
    main()
