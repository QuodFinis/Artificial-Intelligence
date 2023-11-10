import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Read the data
df = pd.read_csv("https://www.kaggle.com/code/parulpandey/penguin-dataset-the-new-iris/input")

# Get the shape of the dataset
print("Shape of the dataset: ", df.shape)

# Show the columns of the dataset
print("Columns of the dataset: ", df.columns)

# Show the summary of the dataset
print("Summary of the dataset: ")
print(df.describe())

# Check the null values of the dataset
print("Null values in the dataset: ")
print(df.isnull().sum())

# Check missing values of the dataset
print("Missing values in the dataset: ")
print(df.isna().sum())

# Plot the distribution of columns
df.hist(figsize=(10,10))
plt.show()

# Convert 'gender' column to number
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# Deal with missing values
df = df.fillna(df.mean())
