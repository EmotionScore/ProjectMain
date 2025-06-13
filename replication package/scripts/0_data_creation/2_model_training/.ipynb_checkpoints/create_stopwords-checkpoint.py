# create_stopwords.py
# ---------------------------------------------------
# Run this ONLY if this script is saved in /data/
# It will correctly save to replication_package/data/stopwords.pkl
# ---------------------------------------------------

import os
import joblib
from nltk.corpus import stopwords
import nltk

# Download NLTK stopwords
nltk.download('stopwords')

# Get stopwords
base_stopwords = stopwords.words('english')
custom_words = [] 
all_stopwords = list(set(base_stopwords + custom_words))

# ---------------------------------------------------
# Go level UP to get to replication_package
# ---------------------------------------------------
this_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(this_script_dir, '..', '..'))
save_dir = os.path.join(project_root, 'data')
os.makedirs(save_dir, exist_ok=True)

# Save the file
output_path = os.path.join(save_dir, 'stopwords.pkl')
joblib.dump(all_stopwords, output_path)

print(f"\n stopwords.pkl saved to:\n{output_path}")
