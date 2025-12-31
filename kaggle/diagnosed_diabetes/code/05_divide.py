# train.csv의 각 컬럼별 데이터 타입 확인
import pandas as pd

df = pd.read_csv('train.csv')
print('--- train.csv 데이터 타입 ---')
print(df.dtypes)

# object 타입(범주형) 컬럼만 object.csv로 저장


# object 타입(범주형) 컬럼과 id 컬럼을 object.csv로 저장
object_cols = df.select_dtypes(include=['object'])
object_with_id = pd.concat([df[['id']], object_cols], axis=1)
object_with_id.to_csv('object.csv', index=False)
print('\nid 컬럼을 포함한 object 타입(범주형) 컬럼이 object.csv로 저장되었습니다.')

# object가 아닌(수치형 등) 컬럼만 number.csv로 저장
number_cols = df.select_dtypes(exclude=['object'])
number_cols.to_csv('number.csv', index=False)
print('object가 아닌(수치형 등) 컬럼이 number.csv로 저장되었습니다.')
