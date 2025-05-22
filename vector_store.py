import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, embedding_dir):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dir = embedding_dir
        self.index_path = os.path.join(embedding_dir, "index.faiss")
        self.meta_path = os.path.join(embedding_dir, "meta.json")
        self.index = faiss.IndexFlatL2(384)
        self.metadata = []

        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r") as f:
                self.metadata = json.load(f)

    def is_index_built(self):
        return self.index.ntotal > 0

    def build_index(self, chunks):
        embeddings = [self.model.encode(chunk["content"]) for chunk in chunks]
        self.index.add(np.array(embeddings))
        self.metadata = chunks
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w") as f:
            json.dump(self.metadata, f)

    def search(self, query, k=5):
        query_embedding = self.model.encode([query])
        _, indices = self.index.search(np.array(query_embedding), k)
        return [self.metadata[i] for i in indices[0]]
