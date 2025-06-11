
# BASED ON: Emotion and Reason in Political LanguageL: Replication Package
# Gennaro and Ash

# ===============================================
# MODIFIED FOR MASTER'S PROJECT - GUARDIAN DATA
# Description: Adapted to load CSV files with text
#              articles from Guardian (2000–2023)
#              instead of extracting from ZIP files
# ===============================================

###################################
#     Modules                   ###
###################################

import os
import pandas as pd
import joblib
from glob import glob
import re

###################################
# Universal project paths
###################################

# Current script directory: scripts/data_creation/data_preprocessing/
script_dir = os.path.dirname(__file__)

# Project root is 3 levels up
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))

# Input: where raw CSVs are stored
data_path = os.path.join(project_root, 'data', 'relevant_guardian_data')

# Output: where processed data will be saved
output_path = os.path.join(project_root, 'data')
os.makedirs(output_path, exist_ok=True)


###################################
# Load and aggregate CSV files   ##
###################################

file_paths = sorted(glob(os.path.join(data_path, "relevant_texts_*.csv")))
data = []

for file_path in file_paths:
    df = pd.read_csv(file_path)

    for col in ['text', 'web_publication_date']:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in file: {file_path}")

    for idx, row in df.iterrows():
        identifier = f"{os.path.basename(file_path)}_{idx}"
        text = str(row["text"]) if pd.notnull(row["text"]) else ""
        date = row["web_publication_date"]
        data.append([identifier, text, date])

print(f"Loaded {len(data)} texts from {len(file_paths)} files.")

###################################
# Cleaning                       ##
###################################

# CHANGE: Adjusted cleaning functions to handle [id, text, date]

def cleaning1(lista):
    return [[row[0], row[1].replace('\n', ' '), row[2]] for row in lista]

def cleaning2(lista):
    return [[row[0], re.sub(r'(?<=[.,])(?=[^\s])', r' ', row[1]), row[2]] for row in lista]

def cleaning3(lista):
    return [[row[0], re.sub(r"-\s", "", row[1]), row[2]] for row in lista]

def cleaning4(lista):
    return [[row[0], row[1].replace("\\", ""), row[2]] for row in lista]

def dropnull(lista):
    return [row for row in lista if len(row[1].split()) > 0]

data = cleaning1(data)
data = cleaning2(data)
data = cleaning3(data)
data = cleaning4(data)

data = dropnull(data)

print('Text has been cleaned. Data length is {}'.format(len(data)))

###################################
# Save to 4 parts                ##
###################################

quarter = len(data) // 4
chunks = [
    data[:quarter],
    data[quarter:2*quarter],
    data[2*quarter:3*quarter],
    data[3*quarter:]
]

filenames = [
    'rawarticles_indexed1_n.pkl',
    'rawarticles_indexed2_n.pkl',
    'rawarticles_indexed3_n.pkl',
    'rawarticles_indexed4_n.pkl'
]

for chunk, fname in zip(chunks, filenames):
    joblib.dump(chunk, os.path.join(output_path, fname))

print(f"Saved data to 4 files in: {output_path}")



import yfinance as yf
apple = yf.download("AAPL", start="2020-01-01", end="2023-12-31")

