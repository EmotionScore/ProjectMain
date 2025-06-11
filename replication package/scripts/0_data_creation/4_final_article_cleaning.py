
# BASED ON: Emotion and Reason in Political LanguageL: Replication Package
# Gennaro and Ash

# ===============================================
# MODIFIED FOR MASTER'S PROJECT - GUARDIAN DATA
# Description: Removes rare words (frequency < 10)
#              from the final cleaned documents.
# ===============================================

import os
import joblib
from multiprocessing import Pool, freeze_support

###################################
# Setup paths                    ##
###################################

# Get current script location and repo root
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))

# Folder with input .pkl files
data_path = os.path.join(project_root, 'data')

# Load word frequency counter (from step 3)
word_counts = joblib.load(os.path.join(project_root, 'data', 'word_frequencies', 'word_counts.pkl'))

###################################
# Define cleaning function       ##
###################################

# Remove words that appear less than 10 times globally
def remove_rare_words(data_list):
    cleaned = []
    for row in data_list:
        filtered_words = [word for word in row[1] if word_counts[word] >= 10]
        cleaned.append([row[0], filtered_words, row[2]])
    return cleaned

def final_cleaning(file_path):
    data = joblib.load(file_path)
    data = remove_rare_words(data)
    output_file = file_path.replace('_temp.pkl', '_clean.pkl')
    joblib.dump(data, output_file)

###################################
# List of files to process       ##
###################################

files = [
    'rawarticles_indexed1_n_temp.pkl',
    'rawarticles_indexed2_n_temp.pkl',
    'rawarticles_indexed3_n_temp.pkl',
    'rawarticles_indexed4_n_temp.pkl'
]

files = [[os.path.join(data_path, f)] for f in files]

###################################
# Run in parallel                ##
###################################

def main():
    with Pool(4) as pool:
        pool.starmap(final_cleaning, files)

if __name__ == "__main__":
    freeze_support()
    main()
