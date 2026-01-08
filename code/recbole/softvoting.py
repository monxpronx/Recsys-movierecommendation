import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# ==========================================
# 설정 (가중치 조절)
# ==========================================
FILES = {
    'sasrec': 'SASRec.csv',
    'ease': 'EASE.csv',
    'lightgcn': 'LightGCN.csv'
}

# 가중치 합은 1이 되도록 설정 (성능 좋은 모델에 더 많이 부여)
WEIGHTS = {
    'sasrec': 0.5,
    'ease': 0.3,
    'lightgcn': 0.2
}

OUTPUT_FILE = 'submission_softvoting.csv'

def load_and_normalize(filepath, model_name):
    print(f"Loading {model_name} from {filepath}...")
    df = pd.read_csv(filepath)
    
    # 컬럼 이름 통일 (score -> {model_name}_score)
    df.rename(columns={'score': f'{model_name}_score'}, inplace=True)
    
    # Min-Max Scaling (점수 범위를 0~1로 맞춤)
    scaler = MinMaxScaler()
    df[f'{model_name}_score'] = scaler.fit_transform(df[[f'{model_name}_score']])
    return df

def main():
    # 1. 데이터 로드 및 정규화
    dfs = []
    for name, path in FILES.items():
        if os.path.exists(path):
            dfs.append(load_and_normalize(path, name))
        else:
            print(f"[Warning] File not found: {path}. Skipping...")

    if not dfs:
        print("No files loaded.")
        return

    # 2. 데이터 병합 (Outer Join)
    # 어떤 모델은 추천했지만 어떤 모델은 추천 안 한 경우도 포함
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=['user', 'item'], how='outer')

    # 3. 결측치(NaN) 처리
    # 특정 모델이 추천 안 한 아이템은 점수 0점 처리
    merged_df.fillna(0, inplace=True)

    # 4. Soft Voting (Weighted Sum)
    print("Calculating weighted scores...")
    merged_df['final_score'] = 0
    
    for name, weight in WEIGHTS.items():
        col_name = f'{name}_score'
        if col_name in merged_df.columns:
            merged_df['final_score'] += merged_df[col_name] * weight

    # 5. 최종 Top-10 추출
    print("Sorting and selecting Top-10...")
    # 유저별로 점수 내림차순 정렬
    merged_df.sort_values(['user', 'final_score'], ascending=[True, False], inplace=True)
    
    # 상위 10개만 남기기
    final_submission = merged_df.groupby('user').head(10)[['user', 'item']]

    # 6. 저장
    final_submission.to_csv(OUTPUT_FILE, index=False)
    print(f"[Done] Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()