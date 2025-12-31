# 데이터 탐색: train.csv 구조, 통계, 결측치, 샘플 확인
import pandas as pd

train = pd.read_csv('train.csv')

print('--- train.csv info() ---')
print(train.info())

print('\n--- train.csv describe() ---')
print(train.describe(include='all'))

print('\n--- train.csv 결측치 현황 ---')
print(train.isnull().sum())

print('\n--- train.csv 샘플 (head) ---')
print(train.head())

# 추가 데이터 이해 및 탐색
print('\n--- diagnosed_diabetes 분포 ---')
print(train['diagnosed_diabetes'].value_counts(normalize=True))

print('\n--- 주요 범주형 변수 분포 ---')
cat_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']
for col in cat_cols:
	print(f'\n[{col}] 분포:')
	print(train[col].value_counts())

print('\n--- 수치형 변수 상관관계 (상위 10개) ---')
cor = train.corr(numeric_only=True)['diagnosed_diabetes'].abs().sort_values(ascending=False)
print(cor[1:11])

# 각 컬럼별 diagnosed_diabetes와의 상관계수 출력
print('\n--- diagnosed_diabetes와 각 수치형 변수의 상관계수 ---')
cor_all = train.corr(numeric_only=True)['diagnosed_diabetes']
for col, val in cor_all.items():
	if col != 'diagnosed_diabetes':
		print(f'{col}: {val:.6f}')

# 상관관계 0.05 이하인 수치형 변수 제거 후 train2.csv로 저장
threshold = 0.05
drop_cols = [col for col, val in cor_all.items() if abs(val) <= threshold and col != 'diagnosed_diabetes']
print(f'\n--- 상관관계 0.05 이하 제거 컬럼: {drop_cols} ---')
train2 = train.drop(columns=drop_cols)
# id 컬럼도 제외 (모델 학습에 불필요)
if 'id' in train2.columns:
	train2 = train2.drop(columns=['id'])
train2.to_csv('train2.csv', index=False)
print('\ntrain2.csv 저장 완료!')
