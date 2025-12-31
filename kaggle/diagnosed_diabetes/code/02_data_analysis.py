# train2.csv 데이터 분포, 이상치, 결측치 확인
import pandas as pd
import numpy as np

df = pd.read_csv('train2.csv')

print('--- train2.csv info() ---')
print(df.info())

print('\n--- train2.csv describe() ---')
print(df.describe(include='all'))

print('\n--- train2.csv 결측치 현황 ---')
print(df.isnull().sum())

# 이상치 탐지 (IQR 방식)
print('\n--- train2.csv 이상치 개수 (수치형 변수별) ---')
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
	Q1 = df[col].quantile(0.25)
	Q3 = df[col].quantile(0.75)
	IQR = Q3 - Q1
	outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
	print(f'{col}: {outliers}')

# 박스플롯 시각화 (수치형 변수별)
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs('boxplots', exist_ok=True)
for col in num_cols:
	plt.figure(figsize=(6, 4))
	sns.boxplot(x=df[col], orient='h')
	plt.title(f'Boxplot of {col}')
	plt.tight_layout()
	plt.savefig(f'boxplots/boxplot_{col}.png')
	plt.close()
print('\n박스플롯 이미지가 boxplots 폴더에 저장되었습니다.')
