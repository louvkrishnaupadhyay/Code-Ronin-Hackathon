import numpy as np
import os
import pickle
import sys

# Quantum Enhancement: Try to load hybrid search module
# Catch ALL exceptions so that any PennyLane/sklearn issue on this machine
# never prevents the Database class from being imported.
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.quantum_similarity import hybrid_search as _quantum_hybrid_search
    QUANTUM_MODE = True
    print("⚡ Quantum-Hybrid Similarity Engine Active")
except Exception as e:
    QUANTUM_MODE = False
    _quantum_hybrid_search = None
    print(f"📌 Classical Cosine Engine Active (Quantum unavailable: {e})")

class Database:
    def __init__(self, embedding_dim=768, db_path='data/database.pkl'):
        """Initializes the vector database for similarity search based on a dictionary format."""
        self.entry_dim = embedding_dim
        self.db_path = db_path
        
        # Ensure data folder exists
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        
        # Internal Storage: { "label1": [embedding1, embedding2], "label2": [embedding3] }
        self.knowledge = {}
        
        # Load existing data from disk (as requested: via pickling a dictionary)
        self.load_db()

    def add_entry(self, embedding, label):
        """Adds a new object embedding to the dictionary grouped by class label and saves to disk."""
        # Ensure embedding is 1D vector of correct size
        vector = embedding.flatten().astype('float32')
        
        if label not in self.knowledge:
            self.knowledge[label] = []
        
        self.knowledge[label].append(vector)
        
        # Persist updated DB
        self.save_db()
        
        total_samples = sum(len(embs) for embs in self.knowledge.values())
        print(f"Added new sample for {label}. Total samples across {len(self.knowledge)} classes: {total_samples}")

    def search_entry(self, embedding, threshold=0.7):
        """
        Searches for the most similar embedding.
        Uses Quantum-Hybrid similarity if available, else falls back to classical cosine.
        Returns: (label, confidence_score) or ("Unknown", score)
        """
        if not self.knowledge:
            return "Unknown", 0.0

        # --- 9. INTEGRATION POINT: Use Quantum Hybrid if available ---
        if QUANTUM_MODE:
            try:
                return _quantum_hybrid_search(embedding, self.knowledge, threshold)
            except Exception as e:
                print(f"[Quantum Search] Error, falling back to cosine: {e}")

        # --- Classical Fallback ---
        vector = embedding.flatten().astype('float32')
        best_score = -1.0
        best_label = "Unknown"
        
        for label, embeddings in self.knowledge.items():
            for stored_vec in embeddings:
                sim = np.dot(vector, stored_vec)
                if sim > best_score:
                    best_score = float(sim)
                    best_label = label
                    
        if best_score >= threshold:
            return best_label, best_score
        else:
            return "Unknown", best_score

    def save_db(self):
        """Saves embeddings structure persistently using a pickle file."""
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.knowledge, f)
        except Exception as e:
            print(f"Failed to save database: {e}")

    def load_db(self):
        """Loads embeddings from the pickle file."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    self.knowledge = pickle.load(f)
                
                # Verify that all embeddings match the expected dimension
                for label, embeddings in list(self.knowledge.items()):
                    valid_embs = [emb for emb in embeddings if emb.shape[0] == self.entry_dim]
                    self.knowledge[label] = valid_embs
                    
                print(f"Loaded DB: {sum(len(e) for e in self.knowledge.values())} embeddings across {len(self.knowledge)} classes.")
            except Exception as e:
                print(f"Error loading database {self.db_path}: {e}")
                self.knowledge = {}
        else:
            print(f"Database {self.db_path} not found. Starting completely fresh.")

    @property
    def labels(self):
        # Compatibility property for the Streamlit UI to display total unique objects
        return list(self.knowledge.keys())

    @property
    def vectors_path(self):
        # Compatibility property for Streamlit UI reset functionality
        return self.db_path

    @property
    def labels_path(self):
        # Compatibility property for Streamlit UI reset functionality
        return self.db_path
