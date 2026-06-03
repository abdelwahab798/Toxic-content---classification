import re 
import json
import pickle
import torch

device=torch.device("cuda" if torch.cuda.is_available else "cpu")
with open(r'D:\__Projects\Toxic-content---classification\app\config.json', 'r') as f:
    config = json.load(f)

with open(r'D:\__Projects\Toxic-content---classification\app\modelsvocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

def clean_text(text):
    text=text.lower()
    text=re.sub(r'\d+', '', text)
    text=re.sub(r'[^\w\s]', '', text)
    return text

def encode(text):
    tokens = clean_text(text).split()[:config['max_length']]
    encoded = [vocab.get(token, vocab.get("<oov>", 1)) for token in tokens]
    encoded += [vocab.get("<pad>", 0)] * (config['max_length'] - len(encoded))
    return torch.tensor([encoded], dtype=torch.long).to(device)