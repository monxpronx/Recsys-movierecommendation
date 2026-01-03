import pandas as pd
import pickle
import os

def save_user_history(train_csv_path, output_path='user_history.pkl'):
    """
    train_ratings.csv를 읽어 시간순으로 정렬된 유저별 시청 시퀀스를 저장 (원본 ID 기준)
    """
    print(f"📂 {train_csv_path} 로드 및 처리 중...")
    
    # 1. 데이터 로드
    df = pd.read_csv(train_csv_path)
    
    # 2. 안전장치: ID를 문자열로 통일 (나중에 매핑 오류 방지)
    df['user'] = df['user'].astype(str)
    df['item'] = df['item'].astype(str)
    
    # 3. 시간 순서 정렬 (Sequential 모델의 핵심!)
    if 'time' in df.columns:
        df = df.sort_values(by=['user', 'time'])
    else:
        print("⚠️ 'time' 컬럼이 없어 데이터 순서대로 시퀀스를 생성합니다.")
    
    # 4. 유저별 아이템 리스트 그룹화 (원본 ID 그대로 저장)
    history_dict = df.groupby('user')['item'].apply(list).to_dict()
    
    # 5. 파일 저장
    with open(output_path, 'wb') as f:
        pickle.dump(history_dict, f)
        
    print(f"✅ 시청 내역 저장 완료! ({output_path}) - 총 유저: {len(history_dict)}")

if __name__ == "__main__":
    save_user_history('../../data/train/train_ratings.csv', 'user_history.pkl')