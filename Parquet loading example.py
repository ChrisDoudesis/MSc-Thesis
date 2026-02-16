import pandas as pd

# df = pd.read_parquet('house-price.parquet')
# df.head()
# df.info()
# df.describe()

# print(df.columns)

# import os
# print(os.getcwd())
import os
from pathlib import Path

print("cwd:", os.getcwd())                 # folder the IDE runs from
print("__file__:", Path(__file__).resolve())  # full path of this script
print("script dir:", Path(__file__).resolve().parent)
