import pandas as pd

patient_data_path = 'data/patient_data.csv'
health_data_path= 'data/healthy_data.csv'
# Load the first CSV file into a pandas DataFrame
df1 = pd.read_csv(patient_data_path)

# Load the second CSV file into a pandas DataFrame
df2 = pd.read_csv(health_data_path)
print(len(df2))
print(len(df1))

# Concatenate the two DataFrames along the rows
df = pd.concat([df1, df2], axis=0)

# Save the concatenated DataFrame to a new CSV file
df.to_csv("data/train.csv",encoding='utf_8_sig', index=False)

