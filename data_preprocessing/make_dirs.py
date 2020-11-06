'''
Utility function to make directories 
by subject id.
'''


import os 
import pandas as pd
df=pd.read_csv('/no_backups/g009/data/oasis1_3_labels.csv')
ids=list(df['MR ID'])
for id in ids:
	os.system('mkdir '+id)


