# BASED ON: Emotion and Reason in Political LanguageL: Replication Package
# Gennaro and Ash

# ===============================================
# MODIFIED FOR MASTER'S PROJECT - GUARDIAN DATA
# Description: Preprocesses cleaned Guardian texts:
#              tokenization, stemming, POS tagging,
#              stopword removal.
# ===============================================

import os
import joblib
from string import punctuation
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk import download
from multiprocessing import Pool, freeze_support
from nltk.corpus import stopwords

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

######################
# Setup paths       ##
######################

# Get current script directory
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))

# Input/output directories
data_path = os.path.join(project_root, 'data')

# Filenames to process
DATI = [
    'rawarticles_indexed1_n.pkl',
    'rawarticles_indexed2_n.pkl',
    'rawarticles_indexed3_n.pkl',
    'rawarticles_indexed4_n.pkl'
]

DATI = [[os.path.join(data_path, fname)] for fname in DATI]

###################################
# Preprocessing functions        ##
###################################

translator = str.maketrans('', '', punctuation)
stemmer = SnowballStemmer("english")
tagger = nltk.perceptron.PerceptronTagger()

# Remove punctuation
def pro1(lista):
    return [[row[0], row[1].translate(translator), row[2]] for row in lista]

# Tokenize using NLTK (replaces gensim.simple_preprocess)
def pro2(lista):
    return [[row[0], word_tokenize(row[1].lower()), row[2]] for row in lista]

# Remove digits
def pro3(lista):
    return [[row[0], [w for w in row[1] if not w.isdigit()], row[2]] for row in lista]

# Remove short words (<=2 chars)
def pro4(lista):
    return [[row[0], [w for w in row[1] if len(w) > 2], row[2]] for row in lista]

# Keep only nouns, verbs, adjectives
def tags(lista):
    tagged = [[row[0], nltk.pos_tag(row[1]), row[2]] for row in lista]
    filtered = [
        [row[0], [w for w, tag in row[1] if tag.startswith(('N', 'V', 'J'))], row[2]]
        for row in tagged
    ]
    return filtered

# Stemming
def pro5(lista):
    return [[row[0], [stemmer.stem(word) for word in row[1]], row[2]] for row in lista]

# Remove stopwords
all_stopwords = set(stopwords.words('english'))

def pro6(lista):
    return [[row[0], [w for w in row[1] if w not in all_stopwords], row[2]] for row in lista]

# Remove empty processed texts
def dropnull(lista):
    return [row for row in lista if len(row[1]) > 0]

###################################
# Main function                  ##
###################################

def preprocessing(data_name):
    data = joblib.load(data_name)
    data = pro1(data)
    data = pro2(data)
    data = pro3(data)
    data = pro4(data)
    data = tags(data)
    data = pro5(data)
    data = pro6(data)
    data = dropnull(data)

    output_name = data_name.replace('.pkl', '_temp.pkl')
    joblib.dump(data, output_name)

###################################
# Run with multiprocessing       ##
###################################

def main():
    for args in DATI:
        preprocessing(*args)
if __name__ == "__main__":
    freeze_support()
    main()
