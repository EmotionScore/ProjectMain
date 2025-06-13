# Emotion and Reason in Political Language: Replication Package
# Gennaro and Ash

# Description:
# - Extract sentences from the corpus

###################################
#     Modules                   ###
###################################

import os
from random import shuffle
import nltk
from nltk.tokenize import sent_tokenize
from nltk.stem.snowball import SnowballStemmer
import joblib
from multiprocessing import Pool, freeze_support

nltk.download('punkt')
# Set correct absolute path to the data folder
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '../../../data')
os.chdir(data_path)


tagger = nltk.perceptron.PerceptronTagger()
stemmer = SnowballStemmer("english")

###################################
#     Working Directory         ###
###################################


stopwords = joblib.load('stopwords.pkl')
word_counts_path = os.path.join('word_frequencies', 'word_counts.pkl')
count = joblib.load(word_counts_path)

###################################
#     Extract sentences         ###
###################################

def extract_sentences(dataname):
    data = joblib.load(dataname)
    data = [a[1] for a in data] 

    sentences = []
    
    for doc in data:
        try:
            if isinstance(doc, list):
                doc = ' '.join(doc)
            doc_sentences = sent_tokenize(doc)
            sentences += doc_sentences
        except Exception as e:
            print(f"Error tokenizing: {e}")

    # Drop empty/short sentences
    sentences = [s for s in sentences if len(s.split()) > 1]

    # Tokenize + clean
    sentences = [nltk.word_tokenize(s.lower()) for s in sentences]
    sentences = [[a for a in s if not a.isdigit()] for s in sentences]
    sentences = [[a for a in s if len(a) > 2] for s in sentences]

    # POS tagging and filtering
    sentences = [tagger.tag(s) for s in sentences]
    sentences = [[i[0] for i in s if i[1].startswith(('N', 'V', 'J'))] for s in sentences]

    # Stemming
    sentences = [[stemmer.stem(i) for i in s] for s in sentences]

    # Remove stopwords and infrequent words
    sentences = [[a for a in b if a not in stopwords] for b in sentences]
    sentences = [[a for a in b if count.get(a, 0) >= 10] for b in sentences]

    # Final cleanup
    sentences = [s for s in sentences if len(s) > 1]
    shuffle(sentences)

    lab = dataname.replace('rawarticles_', 'sentences_')
    print(f'{dataname} processed')
    joblib.dump(sentences, lab)
    print(f'{lab} saved')

###################################
#      Multiprocessing          ###
###################################

# Use actual temp files
DATI = [
    os.path.join(data_path, 'rawarticles_indexed1_n_temp.pkl'),
    os.path.join(data_path, 'rawarticles_indexed2_n_temp.pkl'),
    os.path.join(data_path, 'rawarticles_indexed3_n_temp.pkl'),
    os.path.join(data_path, 'rawarticles_indexed4_n_temp.pkl')
]
DATI = [[a] for a in DATI]


def main():
    with Pool(4) as pool:
        pool.starmap(extract_sentences, DATI)

if __name__ == "__main__":
    freeze_support()
    main()
