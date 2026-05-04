import sys
# Block broken TensorFlow BEFORE any imports
sys.modules['tensorflow'] = None
sys.modules['tensorflow.python'] = None

import json, time, os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def encode(texts, model_name='sentence-transformers/all-MiniLM-L6-v2', batch_size=256):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True,
                           max_length=128, return_tensors='pt')
        if torch.cuda.is_available():
            encoded = {k: v.cuda() for k, v in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
        mask = encoded['attention_mask'].unsqueeze(-1).float()
        embs = (output.last_hidden_state * mask).sum(1) / mask.sum(1)
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        all_embs.append(embs.cpu().numpy())
        if i % 5000 == 0:
            print(f"  {i:,}/{len(texts):,}...")
    return np.concatenate(all_embs, axis=0)

features_file = sys.argv[1]
output_path = sys.argv[2]
print(features_file)
print("Loading...")
with open(features_file) as f:
    data = json.load(f)
texts = [item.get('prompt', '')[:2000] for item in data]
print(f"  {len(texts):,} texts")

t0 = time.time()
embeddings = encode(texts)
print(f"  Encoded: {embeddings.shape} [{time.time()-t0:.1f}s]")

np.save(output_path, embeddings)
print(f"  Saved: {output_path}")