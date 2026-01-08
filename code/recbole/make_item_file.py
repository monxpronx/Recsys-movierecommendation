import pandas as pd
import os

# 데이터 경로 설정
DATA_PATH = "../../data/train" 

print("Loading data...")
# TSV 파일 로드
directors = pd.read_csv(os.path.join(DATA_PATH, 'directors.tsv'), sep='\t')
genres = pd.read_csv(os.path.join(DATA_PATH, 'genres.tsv'), sep='\t')
writers = pd.read_csv(os.path.join(DATA_PATH, 'writers.tsv'), sep='\t')
years = pd.read_csv(os.path.join(DATA_PATH, 'years.tsv'), sep='\t')

# ========================================================
# [핵심] Grouping & Joining (공백 치환 없이 바로 합치기)
# ========================================================
def group_features(df, id_col, value_col):
    # 같은 item_id를 가진 행들을 모아서, 값(value)을 공백(' ')으로 이어 붙임
    # 예: 
    # Item 1 | nm001   ->  Item 1 | "nm001 nm002"
    # Item 1 | nm002
    return df.groupby(id_col)[value_col].apply(lambda x: ' '.join(x.astype(str))).reset_index()

print("Grouping features...")
directors_agg = group_features(directors, 'item', 'director')
writers_agg = group_features(writers, 'item', 'writer')
genres_agg = group_features(genres, 'item', 'genre')

# Years는 보통 영화당 1개이므로 중복 제거만
years_agg = years.drop_duplicates(subset=['item'])

# ========================================================
# Merge (하나의 테이블로 합치기)
# ========================================================
print("Merging...")

# 모든 영화 ID 리스트 확보
all_items = pd.concat([
    directors_agg['item'], 
    writers_agg['item'], 
    genres_agg['item']
]).unique()

df_final = pd.DataFrame(all_items, columns=['item'])

# 하나씩 붙이기 (Left Join)
df_final = pd.merge(df_final, genres_agg, on='item', how='left')
df_final = pd.merge(df_final, directors_agg, on='item', how='left')
df_final = pd.merge(df_final, writers_agg, on='item', how='left')
df_final = pd.merge(df_final, years_agg, on='item', how='left')

# 결측치는 "Unknown" 등으로 채움
df_final.fillna("Unknown", inplace=True)

# ========================================================
# 저장 (RecBole 헤더 포맷 적용)
# ========================================================
# 데이터 자체가 깔끔하므로 헤더에 타입만 명시해주면 끝입니다.
rename_map = {
    'item': 'item:token',
    'genre': 'genre:token_seq',      
    'director': 'director:token_seq', 
    'writer': 'writer:token_seq',     
    'year': 'year:token'             
}

df_final.rename(columns=rename_map, inplace=True)

output_file = "movie.item"
df_final.to_csv(output_file, sep='\t', index=False)

print(f"Done! Saved to {output_file}")
print(df_final.head())