# XGBoost를 이용한 당뇨병 예측 (train.csv → test.csv)
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. 데이터 로드
train = pd.read_csv('combined2_2.csv')
test = pd.read_csv('combined2_2_test.csv')

# 2. 범주형 변수 인코딩 (train/test 동일하게)
cat_cols = train.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
	le = LabelEncoder()
	all_vals = pd.concat([train[col], test[col]], axis=0).astype(str)
	le.fit(all_vals)
	train[col] = le.transform(train[col].astype(str))
	test[col] = le.transform(test[col].astype(str))

# 3. Feature/Target 분리
X = train.drop(columns=['id', 'diagnosed_diabetes'])
y = train['diagnosed_diabetes'].astype(int)
X_test = test.drop(columns=['id'])
test_ids = test['id']

# 4. 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 5. XGBoost 학습 (5-Fold CV)
n_splits = 10 # 폴드 수
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

xgb_oof = np.zeros(len(X))
xgb_test = np.zeros(len(X_test))

print('Starting XGBoost Cross-Validation Training...')

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
	X_tr, X_va = X_scaled[train_idx], X_scaled[val_idx]
	y_tr, y_va = y.iloc[train_idx].values, y.iloc[val_idx].values

	clf = XGBClassifier(
		n_estimators=5000, 
		learning_rate=0.02,
		max_depth=6,
		subsample=0.8,
		colsample_bytree=0.8,
		early_stopping_rounds=50,
		eval_metric='auc',
		random_state=42,
		n_jobs=-1,
		tree_method='hist',
		device='cuda'
	)
	clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
	xgb_oof[val_idx] = clf.predict_proba(X_va)[:, 1]
	xgb_test += clf.predict_proba(X_test_scaled)[:, 1] / n_splits
	print(f"Fold {fold+1} completed.")

auc_xgb = roc_auc_score(y, xgb_oof)
print(f"\nXGBoost OOF AUC: {auc_xgb:.4f}")

# 6. 제출 파일 생성 (당뇨병일 확률)
submission = pd.DataFrame({'id': test_ids, 'diagnosed_diabetes': xgb_test})
submission.to_csv('submission_xgb_test_combined2_2_4.csv', index=False)
print("\nXGBoost submission file created: submission_xgb_test_combined2_2_4.csv")

