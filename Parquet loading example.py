import pandas as pd

df = pd.read_parquet('house-price.parquet')
df.head()
df.info()
df.describe()

print(df.columns)