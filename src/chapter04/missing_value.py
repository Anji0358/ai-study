import pandas as pd
from io import StringIO
from sklearn.impute import SimpleImputer
import numpy as np

csv_data = '''
A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,
'''

df = pd.read_csv(StringIO(csv_data))
# print(df)
# print(df.isnull().sum())
imr = SimpleImputer(missing_values=np.nan,strategy='mean')
imr=imr.fit(df.values)
imputed_data=imr.fit_transform(df.values)

# 配列表示
print(imputed_data)
# 表表示
print(df.fillna(df.mean()))

