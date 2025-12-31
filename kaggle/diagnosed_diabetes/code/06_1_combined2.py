# object_encoded2.csv와 number.csv를 id 기준으로 합쳐 combined2.csv로 저장
import pandas as pd

number_df = pd.read_csv('number.csv')
object_encoded2_df = pd.read_csv('object_encoded2.csv')

# id 기준으로 병합
combined2_df = pd.merge(number_df, object_encoded2_df, on='id', how='inner')
combined2_df.to_csv('combined2.csv', index=False)
print('combined2.csv로 저장 완료!')
