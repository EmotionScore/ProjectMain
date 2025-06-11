# Emotion and Reason in Political Language: Replication Package
# Gennaro and Ash

# Description:
# - Extract LIWC dictionaries
# - Match them to NLP vocabulary
# - Eliminate words that are too far from centroid
# Output: final affect and cognition dictionaries and eliminated words

###################################
#     Modules                   ###
###################################
import os
import string
from glob import glob
import spacy
from scipy.spatial import distance
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from nltk.tag import perceptron
from nltk.stem.snowball import SnowballStemmer
import joblib
import re

# Ensure NLTK resources
nltk.download('wordnet', quiet=True)

###################################
#     Fixed Paths               ###
###################################
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.normpath(os.path.join(script_dir, '../../../data'))
liwc_dir = os.path.join(data_dir, 'liwc')
results_dir = os.path.normpath(os.path.join(script_dir, '../../../results'))

# Create directories if missing
os.makedirs(liwc_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

###################################
#     Load SpaCy Model          ###
###################################
try:
    nlp = spacy.load('en_core_web_lg')
except OSError:
    import subprocess
    subprocess.run(['python3', '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
    nlp = spacy.load('en_core_web_sm')

###################################
#     Initialize NLP tools      ###
###################################
tagger = perceptron.PerceptronTagger()
stemmer = SnowballStemmer('english')

###################################
#     Match stems               ###
###################################
liwc_files = glob(os.path.join(liwc_dir, '*.txt'))
labels2words = {}
for lfile in liwc_files:
    label = os.path.splitext(os.path.basename(lfile))[0].split('-')[-1]
    tokens = open(lfile, encoding='utf-8').read().split()
    labels2words[label] = tokens

# Focus on affect and cogproc
mydict = {k: labels2words.get(k, []) for k in ['affect', 'cogproc']}

# Separate tokens and wildcard patterns
def split_tokens_wildcards(word_list):
    tokens, patterns = set(), []
    for w in word_list:
        if '*' in w:
            patterns.append(w.replace('*', '[a-z]*'))
        else:
            tokens.add(w)
    return tokens, patterns

tokens_affect, patterns_affect = split_tokens_wildcards(mydict['affect'])
tokens_cog, patterns_cog = split_tokens_wildcards(mydict['cogproc'])

# Expand patterns via WordNet
def expand_wildcards(patterns):
    matches = set()
    if not patterns:
        return matches
    regex = re.compile('(' + ')|('.join(patterns) + ')')
    for syn in wn.all_synsets():
        w = syn.lemma_names()[0].lower()
        if '_' in w:
            continue
        if regex.match(w):
            matches.add(w)
    return matches

tokens_affect |= expand_wildcards(patterns_affect)
tokens_cog |= expand_wildcards(patterns_cog)

# Remove punctuation tokens
tokens_affect = {w for w in tokens_affect if not set(w) & set(string.punctuation)}
tokens_cog = {w for w in tokens_cog if not set(w) & set(string.punctuation)}

# Remove intersection
common = tokens_affect & tokens_cog
tokens_affect -= common
tokens_cog -= common

# Keep only tokens in spaCy vocab
tokens_affect = [w for w in tokens_affect if w in nlp.vocab]
tokens_cog = [w for w in tokens_cog if w in nlp.vocab]

###################################
# Eliminate unrelated words     ###
###################################
def find_unrelated(tokens):
    if not tokens:
        return []
    doc = nlp(' '.join(tokens))
    center = doc.vector
    dist_list = []
    for i, token in enumerate(tokens):
        if doc[i].has_vector:
            dist_list.append((token, distance.cosine(center, doc[i].vector)))
    if not dist_list:
        return []
    df = pd.DataFrame(dist_list, columns=['token', 'distance'])
    if df.empty:
        return []
    cutoff = np.percentile(df['distance'], 75)
    return sorted(df.loc[df['distance'] < cutoff, 'token'].tolist())

final_affect = find_unrelated(tokens_affect)
final_cog = find_unrelated(tokens_cog)

# Words eliminated
elim_affect = set(tokens_affect) - set(final_affect)
elim_cog = set(tokens_cog) - set(final_cog)

###################################
#     Post-processing           ###
###################################
# POS filter and stemming
tag_aff = tagger.tag(final_affect)
tag_cog = tagger.tag(final_cog)
final_affect = {stemmer.stem(w) for w, pos in tag_aff if pos.startswith(('N','V','J'))}
final_cog = {stemmer.stem(w) for w, pos in tag_cog if pos.startswith(('N','V','J'))}

# Remove stopwords and procedural words
stop = set()

# Load stopwords if available
stop_path = os.path.join(data_dir, 'stopwords.pkl')
if os.path.exists(stop_path):
    stop = set(joblib.load(stop_path))
else:
    print(f"Warning: stopwords file not found at {stop_path}")

# Load procedural words if available
proc_path = os.path.join(data_dir, 'procedural_words.pkl')
if os.path.exists(proc_path):
    stop |= set(joblib.load(proc_path))
else:
    print(f"Note: procedural words file not found at {proc_path}, skipping")

final_affect -= stop
final_cog -= stop

###################################
#     Save Outputs              ###              ###
###################################
dict_dir = os.path.join(results_dir, 'dictionaries')
os.makedirs(dict_dir, exist_ok=True)
joblib.dump(list(final_cog), os.path.join(dict_dir, 'dictionary_cognition.pkl'))
joblib.dump(list(final_affect), os.path.join(dict_dir, 'dictionary_affect.pkl'))

# Text files
with open(os.path.join(dict_dir, 'dictionary_affect.txt'), 'w') as f:
    f.write(','.join(final_affect))
with open(os.path.join(dict_dir, 'dictionary_cognition.txt'), 'w') as f:
    f.write(','.join(final_cog))

# Eliminated lists
with open(os.path.join(results_dir, 'eliminated_words_affect.txt'), 'w') as f:
    f.write(','.join(elim_affect))
with open(os.path.join(results_dir, 'eliminated_words_cognition.txt'), 'w') as f:
    f.write(','.join(elim_cog))

print("LIWC dictionaries created and saved.")
