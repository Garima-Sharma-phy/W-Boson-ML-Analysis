import pandas as pd
from sklearn.utils import shuffle

print("Loading all CSV files...")

# Load the 4 W-file CSVs (these have both 1s and 0s)
df_w1 = pd.read_csv('w1_ml_ready.csv')
df_w2 = pd.read_csv('w2_ml_ready.csv')

# Load the 2 pure background CSVs (these only have 0s)
df_z = pd.read_csv('z_background_ready.csv')
df_top = pd.read_csv('ttbar_ready.csv')

# IMPORTANT: Ensure the label column names match perfectly!
# Your v3 script calls it 'label', but my script called it 'training_label'. 
# Let's rename the background ones so they match your signal files:
df_z.rename(columns={'training_label': 'label'}, inplace=True)
df_top.rename(columns={'training_label': 'label'}, inplace=True)

# Merge them all into one colossal dataset
print("Merging and shuffling...")
df_master = pd.concat([df_w1, df_w2, df_z, df_top], ignore_index=True)

# Shuffle the deck so the ML model doesn't memorize the order
df_master = shuffle(df_master, random_state=42).reset_index(drop=True)

print(f"Total events in Master Dataset: {len(df_master)}")
df_master.to_csv('final_master_training_data.csv', index=False)
print("Saved as 'final_master_training_data.csv'")