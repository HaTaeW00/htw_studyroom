# number.csv와 object_encoded.csv를 id 기준으로 합쳐 combined.csv로 저장
import pandas as pd

number_df = pd.read_csv('number.csv')
object_encoded_df = pd.read_csv('object_encoded.csv')

# id 기준으로 병합
combined_df = pd.merge(number_df, object_encoded_df, on='id', how='inner')
combined_df.to_csv('combined.csv', index=False)
print('combined.csv로 저장 완료!')
