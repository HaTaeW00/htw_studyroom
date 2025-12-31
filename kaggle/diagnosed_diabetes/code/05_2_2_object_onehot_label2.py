# object.csv 인코딩: 순서형은 OrdinalEncoder, 나머지는 OneHotEncoder(drop='first')
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

df = pd.read_csv('object.csv')

# id는 따로 저장
id_col = df[['id']]

# 순서형 변수 정의 및 순서 지정
ordinal_cols = ['education_level', 'income_level']
ordinal_maps = {
	'education_level': ['No formal', 'Highschool', 'Graduate', 'Postgraduate'],
	'income_level': ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High']
}

# 순서형 인코딩
ordinal_encoder = OrdinalEncoder(categories=[ordinal_maps[col] for col in ordinal_cols])
ordinal_encoded = ordinal_encoder.fit_transform(df[ordinal_cols])
ordinal_df = pd.DataFrame(ordinal_encoded, columns=ordinal_cols)

# 명목형(순서 없는) 변수 인코딩 (drop='first'로 완전 다중공선성 방지)
nominal_cols = ['gender', 'ethnicity', 'smoking_status', 'employment_status']
onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
onehot_encoded = onehot_encoder.fit_transform(df[nominal_cols])
onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(nominal_cols))

# 결과 합치기
encoded_df = pd.concat([id_col, ordinal_df, onehot_df], axis=1)
encoded_df.to_csv('object_encoded2.csv', index=False)
print('object_encoded2.csv로 인코딩 결과가 저장되었습니다.')