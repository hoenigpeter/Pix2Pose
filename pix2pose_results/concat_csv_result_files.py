import os
import os.path as osp
import sys
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

# set the path of the folder where the csv files are located
folder_path = os.path.join(osp.abspath(osp.dirname(__file__)), 'itodd')

# get a list of all csv files in the folder
file_pattern = os.path.join(folder_path, '*.csv')
csv_files = glob.glob(file_pattern)

# combine all csv files into a single dataframe
print('Combining CSV files...')
dfs = []
for csv_file in tqdm(csv_files):
    df = pd.read_csv(csv_file)
    dfs.append(df)
combined_df = pd.concat(dfs, ignore_index=True)

# Set all "time" values to 0.2
time_col_index = combined_df.columns.get_loc("time")
combined_df.iloc[:, time_col_index] = 0.2

# write the modified dataframe to a csv file
output_path = os.path.join(folder_path, 'concatenated_result_files.csv')
print(f'Writing results to {output_path}')
combined_df.to_csv(output_path, index=False)

print('Done!')
