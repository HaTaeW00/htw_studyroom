# combined2.csv에서 diagnosed_diabetes와 각 변수의 상관계수 출력 및 시각화
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('combined2.csv')

# diagnosed_diabetes와의 상관계수 계산 (id, diagnosed_diabetes 제외)
target = 'diagnosed_diabetes'
exclude_cols = ['id', target]
correlations = df.corr(numeric_only=True)[target].drop(exclude_cols, errors='ignore')

print('--- diagnosed_diabetes와 각 변수의 상관계수 ---')
print(correlations)

# 시각화 (막대그래프)
plt.figure(figsize=(10, 6))
correlations.sort_values(ascending=False).plot(kind='bar')
plt.title('Correlation with diagnosed_diabetes')
plt.ylabel('Correlation coefficient')
plt.tight_layout()
plt.savefig('09_check.png')
plt.close()
print('09_check.png로 상관계수 그래프가 저장되었습니다.')
