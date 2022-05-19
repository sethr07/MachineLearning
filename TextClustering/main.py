#!/usr/bin/python3

import pandas as pd 

file = "prob_soln_dump.csv"

df = pd.read_csv(file, encoding= 'unicode_escape')

# print(df.head())
# print(df.info())
# print(df.describe())

"""
Appy NLP and clustering techniques to cluster the problem statements (Title). 
Try different methods and different number of clusters to see best clustering options 
(you may have to write the cluster information back to a file and visually examine what clusters 
are being created)
"""

"""
Step 1: Clean up the data. Remove stopwords, puntuations etc. 
"""



