#vendor-neutral conceptual pipeline
import numpy as np

def embed(texts):
    # Replace with your embedding model call
    raise NotImplementedError

def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return oat(np.dot(a, b))

class TinyVectorIndex:
    def __init__(self):
        self.texts = []
        self.vecs = []
    
    def add(self, texts):
        v = embed(texts)
        self.texts.extend(texts)
        self.vecs.extend(v)
    
    def search(self, query, k=5):
        qv = embed([query])[0]
        scores = [(cosine(qv, v), t) for v, t in zip(self.vecs, self.texts)]
        scores.sort(reverse=True, key=lambda x: x[0])
        return scores[:k]

def build_prompt(query, retrieved):
    ctx = "".join([f"- {t}" for _, t in retrieved])
    return (
        "You are a helpful assistant."
        ""
        "Answer using ONLY the context. If the context is insuffcient, say: I don't know"
        f"Question: {query}"
        f"Context:{ctx}"
        "Answer:"
    )