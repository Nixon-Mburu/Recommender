import pandas as pd
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'movie.csv')

df = pd.read_csv(csv_path)

df = df.drop_duplicates()

df = df.fillna({
    'numeric_columns': 0,
    'string_columns': 'Unknown'
})

if 'release_date' in df.columns:
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

df = df.dropna(how='all')

df = df.reset_index(drop=True)

df.to_csv('new_movie.csv', index=False)

print("Data cleaning completed. Cleaned file saved as 'new_movie.csv'")
