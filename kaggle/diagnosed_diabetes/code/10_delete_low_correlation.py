# combined2.csv에서 diagnosed_diabetes와 상관계수 0.05 미만인 칼럼 제외 후 저장
import pandas as pd

df = pd.read_csv('combined2.csv')

# diagnosed_diabetes와의 상관계수 계산 (id, diagnosed_diabetes 제외)
target = 'diagnosed_diabetes'
exclude_cols = ['id', target]
correlations = df.corr(numeric_only=True)[target].drop(exclude_cols, errors='ignore')

# 상관계수 0.05 미만인 칼럼 찾기
low_corr_cols = correlations[correlations.abs() < 0.05].index.tolist()
print('상관계수 0.05 미만 칼럼:', low_corr_cols)

# 해당 칼럼 제거
df2 = df.drop(columns=low_corr_cols)
df2.to_csv('combined2_2.csv', index=False)
print('combined2_2.csv로 저장 완료!')
