
# BASED ON: Emotion and Reason in Political LanguageL: Replication Package
# Gennaro and Ash

# ===============================================
# MODIFIED FOR MASTER'S PROJECT - GUARDIAN DATA
# Description: Count word frequencies and compute
#              weighted (SIF-style) frequencies.
# ===============================================

import os
import joblib
from collections import Counter

###################################
# Setup paths                    ##
###################################

# Get current script dir and repo root
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))

# Input folder: preprocessed data
data_path = os.path.join(project_root, 'data')

# Output folder
output_path = os.path.join(project_root, 'data', 'word_frequencies')
os.makedirs(output_path, exist_ok=True)

# Files to process
DATI = [
    'rawarticles_indexed1_n_temp.pkl',
    'rawarticles_indexed2_n_temp.pkl',
    'rawarticles_indexed3_n_temp.pkl',
    'rawarticles_indexed4_n_temp.pkl'
]

###################################
# Count word frequencies         ##
###################################

def find_frequencies(file_path):
    data = joblib.load(file_path)
    all_words = [word for row in data for word in row[1]]
    return Counter(all_words)

total_freqs = Counter()
for fname in DATI:
    path = os.path.join(data_path, fname)
    counts = find_frequencies(path)
    total_freqs.update(counts)

# Save absolute frequencies
joblib.dump(total_freqs, os.path.join(output_path, 'word_counts.pkl'))
print(f"Saved absolute frequencies. Vocabulary size: {len(total_freqs)}")

###################################
# Compute weighted frequencies   ##
# SIF-style: a / (a + freq / total)
###################################

a = 0.001
total_count = sum(total_freqs.values())
weighted_freqs = {
    word: a / (a + (count / total_count))
    for word, count in total_freqs.items()
}

# Save weighted frequencies
joblib.dump(weighted_freqs, os.path.join(output_path, 'word_freqs.pkl'))
print("Saved weighted frequencies.")
