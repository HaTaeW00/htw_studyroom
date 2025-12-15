import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import csv  # 추가

def improved_balance_dataset(input_file='processed_data.csv', output_file='improved_balanced_second_ps.csv'):
    """개선된 데이터 밸런싱"""
    print("🎬 개선된 데이터 불균형 해결 시작")
    print("=" * 60)
    
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    df['장르_단일'] = df['장르'].apply(lambda x: x.split(',')[0].strip())
    
    print("📊 현재 장르 분포:")
    genre_counts = df['장르_단일'].value_counts()
    print(genre_counts)
    
    # 1. 더 관대한 최소 샘플 수 (10개 이상)
    min_samples = 10
    valid_genres = genre_counts[genre_counts >= min_samples].index
    df_filtered = df[df['장르_단일'].isin(valid_genres)].copy()
    
    print(f"\n🗑️ {min_samples}개 미만 클래스 제거:")
    removed_genres = set(genre_counts.index) - set(valid_genres)
    for genre in removed_genres:
        print(f"  - {genre}: {genre_counts[genre]}개")
    
    # 2. 적응적 타겟 샘플링 (클래스별로 다른 목표 설정)
    print(f"\n🔄 적응적 데이터 밸런싱 중...")
    
    balanced_dfs = []
    
    for genre in df_filtered['장르_단일'].unique():
        genre_data = df_filtered[df_filtered['장르_단일'] == genre]
        current_count = len(genre_data)
        
        # 클래스별 적응적 목표 샘플 수
        if current_count >= 100:
            target_samples = 80  # 큰 클래스는 80개로
        elif current_count >= 50:
            target_samples = 60  # 중간 클래스는 60개로  
        elif current_count >= 30:
            target_samples = 45  # 작은 클래스는 45개로
        else:
            target_samples = min(35, current_count * 2)  # 매우 작은 클래스는 최대 2배
        
        if current_count > target_samples:
            # 언더샘플링 (랜덤이 아닌 다양성 고려)
            genre_balanced = genre_data.sample(n=target_samples, random_state=42)
            print(f"  📉 {genre}: {current_count} → {target_samples} (언더샘플링)")
        else:
            # 제한적 오버샘플링
            if current_count < target_samples:
                # 원본 + 일부 복제
                original_samples = current_count
                additional_needed = target_samples - current_count
                
                genre_balanced = pd.concat([
                    genre_data,  # 원본 전체
                    genre_data.sample(n=additional_needed, replace=True, random_state=42)  # 추가 샘플
                ])
                print(f"  📈 {genre}: {current_count} → {target_samples} (제한적 오버샘플링)")
            else:
                genre_balanced = genre_data
                print(f"  ✅ {genre}: {current_count} (변경 없음)")
        
        balanced_dfs.append(genre_balanced)
    
    # 결합 및 셔플
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    df_balanced['번호'] = range(1, len(df_balanced) + 1)
    df_balanced = df_balanced.drop('장르_단일', axis=1)
    
    # 저장 (모든 컬럼 쌍따옴표)
    df_balanced.to_csv(output_file, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
    
    print(f"\n📊 개선된 밸런싱 결과:")
    print(f"  원본: {len(df)}개 → 밸런싱: {len(df_balanced)}개")
    
    # 최종 분포 확인
    print(f"\n🎭 최종 장르 분포:")
    final_counts = df_balanced['장르'].apply(lambda x: x.split(',')[0].strip()).value_counts()
    print(final_counts)
    
    return df_balanced

def split_train_evaluation_data_v2(balanced_df, train_file='improved_train_data.csv', eval_file='improved_evaluation_data.csv', test_size=0.2, random_state=42):
    """개선된 데이터 분할"""
    print("\n" + "=" * 60)
    print("📂 개선된 훈련용/평가용 데이터 분할")
    print("=" * 60)
    
    single_genres = balanced_df['장르'].apply(lambda x: x.split(',')[0].strip())
    
    # 8:2 분할
    train_data, eval_data = train_test_split(
        balanced_df,
        test_size=test_size,
        random_state=random_state,
        stratify=single_genres
    )
    
    # 번호 재정렬
    train_data = train_data.copy().reset_index(drop=True)
    eval_data = eval_data.copy().reset_index(drop=True)
    train_data['번호'] = range(1, len(train_data) + 1)
    eval_data['번호'] = range(1, len(eval_data) + 1)
    
    # 저장 (모든 컬럼 쌍따옴표)
    train_data.to_csv(train_file, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
    eval_data.to_csv(eval_file, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
    
    print(f"✅ 개선된 데이터 분할 완료!")
    print(f"📊 분할 결과:")
    print(f"  훈련용: {len(train_data)}개 → {train_file}")
    print(f"  평가용: {len(eval_data)}개 → {eval_file}")
    
    # 장르별 분포
    print(f"\n🎭 훈련용 장르 분포:")
    train_counts = train_data['장르'].apply(lambda x: x.split(',')[0].strip()).value_counts()
    print(train_counts)
    
    print(f"\n🎭 평가용 장르 분포:")
    eval_counts = eval_data['장르'].apply(lambda x: x.split(',')[0].strip()).value_counts()
    print(eval_counts)
    
    return train_data, eval_data

def main():
    """메인 실행 함수"""
    print("🎬 개선된 영화 데이터 전처리 v2")
    print("=" * 60)
    
    try:
        # 개선된 밸런싱
        balanced_df = improved_balance_dataset(
            input_file='processed_data.csv',
            output_file='improved_balanced_second_ps.csv'
        )
        
        # 개선된 분할
        train_data, eval_data = split_train_evaluation_data_v2(
            balanced_df,
            train_file='improved_train_data.csv',
            eval_file='improved_evaluation_data.csv'
        )
        
        print(f"\n🎉 개선된 전처리 완료!")
        print(f"📁 생성된 파일:")
        print(f"  1. improved_balanced_second_ps.csv")
        print(f"  2. improved_train_data.csv ({len(train_data)}개)")
        print(f"  3. improved_evaluation_data.csv ({len(eval_data)}개)")
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()