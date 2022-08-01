import pandas as pd

#

df = pd.read_json("tmp/ib-out/ib-samples.json")

print(df.head())
