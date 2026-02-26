import pandas as pd
from pathlib import Path
# read file at
OUTPUT_PATH = Path("data/results/it/analysis_results.parquet")
# read parquet file
df = pd.read_parquet(OUTPUT_PATH)

# head = df.head()
# print(head)

# filter df to only include rows where symbol is "A2A"
df = df[df['symbol'] == 'A2A.MI']

# count date values in df and sort descendi values 
date_counts = df['date'].value_counts().sort_index(ascending=False)
print(date_counts)

# select from df 'symbol', 'date' and columns ending with "cumul"
# columns = ['symbol', 'date', 'open'] + [col for col in df.columns if col.endswith("cumul")]
# df_selected = df[columns]

# print(df_selected.tail())