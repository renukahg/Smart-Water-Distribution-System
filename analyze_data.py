import pandas as pd
import numpy as np

df = pd.read_csv('c:/Users/Renuka/OneDrive/Desktop/FinalYear/FinalYear/data/leak_data.csv')
print("Value Counts for Leak_Status:")
print(df['Leak_Status'].value_counts())
print("\nCorrelation with Leak_Status:")
print(df.corr()['Leak_Status'].sort_values(ascending=False))
print("\nDescribe data:")
print(df.describe())
