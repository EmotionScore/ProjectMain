# Emotion and Reason in Political Language: Replication Package
# Gennaro and Ash
#
# Description:
# - Model training

import os
import joblib
from gensim.models import Word2Vec

# ─── Paths ────────────────────────────────────────────────────────────
# location of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# data lives here:
wd_data = os.path.normpath(os.path.join(script_dir, '../../../data'))

# model output goes here:
wd_model = os.path.normpath(os.path.join(script_dir, '../../../models'))
os.makedirs(wd_model, exist_ok=True)

# ─── Load Sentence Data ───────────────────────────────────────────────
# no need to cd if you use full paths:
sentence_files = [
    'sentences_indexed1_n_temp.pkl',
    'sentences_indexed2_n_temp.pkl',
    'sentences_indexed3_n_temp.pkl',
    'sentences_indexed4_n_temp.pkl'
]

dataset = []
for fname in sentence_files:
    path = os.path.join(wd_data, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing sentence file: {path}")
    data = joblib.load(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    dataset.extend(data)

if not dataset:
    raise ValueError("Dataset is empty—no sentences to train on.")
print(f"Loaded {len(dataset)} sentences.")

# ─── Model Training ───────────────────────────────────────────────────
# 1) instantiate (no data yet)
w2v = Word2Vec(
    vector_size=300,   # dimensionality
    workers=8,         # threads
    min_count=10,      # ignore infrequent words
    window=8,          # context window
    sample=1e-3        # downsample frequent words
)

# 2) build vocabulary
print("Building vocabulary…")
w2v.build_vocab(dataset, progress_per=10000)
print(f"Vocabulary size: {len(w2v.wv)} tokens.")

# 3) train
print("Training model…")
w2v.train(
    dataset,
    total_examples=w2v.corpus_count,
    epochs=10,
    report_delay=1
)
print("Training complete.")

# ─── Save Model ───────────────────────────────────────────────────────
out_path = os.path.join(wd_model, 'w2v-vectors_8_300.pkl')
w2v.save(out_path)
print(f"Model saved to {out_path}")
