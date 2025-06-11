# Emotion and Reason in Political Language: Replication Package
# Gennaro and Ash

# Description:
# - Find the document vectors for the cognition and affect dictionaries
# - These centroids are SIF weighted averages

###################################
#     Modules                   ###
###################################
import os
import joblib
from gensim.models import Word2Vec
import numpy as np

###################################
#     Fixed Paths               ###
###################################
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.normpath(os.path.join(script_dir, '../../../data'))
model_dir = os.path.normpath(os.path.join(script_dir, '../../../models'))
results_dir = os.path.normpath(os.path.join(script_dir, '../../../results'))
aux_dir = os.path.join(data_dir, '3_auxiliary_data')
os.makedirs(aux_dir, exist_ok=True)

###################################
# Upload all elements           ###
###################################
# Load dictionaries from results folder
dict_dir = os.path.join(results_dir, 'dictionaries')
cog_path = os.path.join(dict_dir, 'dictionary_cognition.pkl')
aff_path = os.path.join(dict_dir, 'dictionary_affect.pkl')

if not os.path.exists(cog_path) or not os.path.exists(aff_path):
    raise FileNotFoundError(f"Dictionary files not found at {cog_path} or {aff_path}")

cognition = joblib.load(cog_path)
affect = joblib.load(aff_path)

# Load word vectors
w2v_path = os.path.join(model_dir, 'w2v-vectors_8_300.pkl')
if not os.path.exists(w2v_path):
    raise FileNotFoundError(f"Word2Vec model not found at {w2v_path}")
w2v = Word2Vec.load(w2v_path, mmap='r')
model = w2v.wv

# Load word frequencies
freqs_path = os.path.join(data_dir, 'word_frequencies', 'word_freqs.pkl')
if not os.path.exists(freqs_path):
    raise FileNotFoundError(f"Word frequencies file not found at {freqs_path}")
freqs = joblib.load(freqs_path)

###################################
# Find the centroid             ###
###################################
def find_centroid(tokens, model, freqs):
    # Compute weighted vectors
    vecs = []
    for w in tokens:
        if w in model and w in freqs:
            vecs.append(model[w] * freqs.get(w, 1))
    # If no vectors found, return zero vector
    if not vecs:
        print(f"Warning: No vectors for tokens {tokens}; returning zero vector.")
        # return a zero vector matching model vector size
        return np.zeros((1, model.vector_size))
    mat = np.vstack(vecs)
    centroid = mat.mean(axis=0).reshape(1, -1)
    return centroid

c_affect = find_centroid(affect, model, freqs)
c_cognition = find_centroid(cognition, model, freqs)

###################################
# Save centroids                ###
###################################
aff_out = os.path.join(aux_dir, 'affect_centroid.pkl')
cog_out = os.path.join(aux_dir, 'cog_centroid.pkl')
joblib.dump(c_affect, aff_out)
joblib.dump(c_cognition, cog_out)

print(f"Saved affect centroid to {aff_out}")
print(f"Saved cognition centroid to {cog_out}")
