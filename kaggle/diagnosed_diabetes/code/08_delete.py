# combined2.csv에서 waist_to_hip_ratio를 제외하고 VIF 계산
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

df = pd.read_csv('combined2.csv')
# id, diagnosed_diabetes, waist_to_hip_ratio 컬럼 제외
drop_cols = [col for col in ['id', 'diagnosed_diabetes', 'waist_to_hip_ratio'] if col in df.columns]
X = df.drop(columns=drop_cols, errors='ignore')

print('--- VIF (Variance Inflation Factor, waist_to_hip_ratio 제외) ---')
vif_data = []
for i in range(X.shape[1]):
	vif = variance_inflation_factor(X.values, i)
	vif_data.append({'feature': X.columns[i], 'VIF': vif})
vif_df = pd.DataFrame(vif_data)
print(vif_df)
