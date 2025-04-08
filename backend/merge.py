import pandas as pd
import os

csv_path = os.path.join(os.path.dirname(__file__), 'movie.csv')

try:
    print(f"Attempting to read file from: {csv_path}")
    df = pd.read_csv(csv_path)

    print("\nColumns in movie.csv:")
    print(df.columns.tolist())

    print("\nFirst few rows of movie.csv:")
    print(df.head())

    print("\nDataframe info:")
    print(df.info())
except FileNotFoundError:
    print(f"Error: Could not find movie.csv at {csv_path}")
    print(f"Current directory contents: {os.listdir(os.path.dirname(__file__))}")
    print("\nPlease ensure movie.csv exists in the backend directory")
