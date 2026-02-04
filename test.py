import torch
import numpy as np
import argparse
import os
import sys
import warnings
from safetensors.torch import load_file

# Suppress the specific torch load warning to keep output clean
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import cosine
except ImportError:
    print("Missing libraries. Please run:")
    print("pip install scikit-learn matplotlib scipy")
    sys.exit(1)

class Word2VecEvaluator:
    def __init__(self, checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}...")
        device = torch.device("cpu") # CPU is sufficient for evaluation
        
        try:
            if checkpoint_path.endswith(".safetensors"):
                # 1. Load weights from safetensors
                state_dict = load_file(checkpoint_path, device="cpu")
                
                # 2. Load vocab from cache (required for safetensors as it only stores weights)
                vocab_paths = [
                    "data/sampled_50m.txt.vocab.pth",
                    checkpoint_path.replace(".safetensors", ".vocab.pth"),
                    "word2vec_sampled_50m.pth" 
                ]
                
                vocab_found = False
                for vp in vocab_paths:
                    if os.path.exists(vp):
                        print(f"Loading vocabulary from: {vp}")
                        vocab_data = torch.load(vp, map_location=device, weights_only=False)
                        self.vocab = vocab_data['vocab']
                        self.idx_to_word = vocab_data['idx_to_word']
                        vocab_found = True
                        break
                
                if not vocab_found:
                    print("Error: Could not find vocabulary file for .safetensors model.")
                    sys.exit(1)
            else:
                # Load legacy .pth checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                self.vocab = checkpoint['vocab']
                self.idx_to_word = checkpoint['idx_to_word']
                state_dict = checkpoint['model_state_dict']
            
            # 3. Extract Weights
            if 'in_embeddings.weight' in state_dict:
                print("Detected Modern Architecture (Negative Sampling).")
                # For Word2Vec, the 'Input' embeddings represent the word meaning best
                self.embeddings = state_dict['in_embeddings.weight'].numpy()
            elif 'embeddings.weight' in state_dict:
                print("Detected Legacy Architecture (Standard CBOW).")
                self.embeddings = state_dict['embeddings.weight'].numpy()
            else:
                print("Error: Could not find embeddings in checkpoint.")
                print(f"Available keys: {list(state_dict.keys())}")
                sys.exit(1)
                
            print(f"Model Loaded. Vocab Size: {len(self.vocab)}, Dimensions: {self.embeddings.shape[1]}")
            
            # 4. Pre-normalize vectors for fast Cosine Similarity
            print("Normalizing vectors for fast search...")
            norm = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norm[norm == 0] = 1e-10 # Avoid division by zero
            self.normalized_embeddings = self.embeddings / norm
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            sys.exit(1)

    def get_vector(self, word):
        if word not in self.vocab:
            return None
        idx = self.vocab[word]
        return self.normalized_embeddings[idx]

    def most_similar(self, word, top_n=10):
        """Finds the N closest words using Dot Product (Cosine Sim)."""
        word = word.lower()
        if word not in self.vocab:
            return None

        target_vector = self.get_vector(word)
        
        # Matrix multiplication: (VocabSize, Dim) x (Dim,) = (VocabSize,)
        similarities = np.dot(self.normalized_embeddings, target_vector)
        
        # Sort descending
        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in sorted_indices:
            # Skip the word itself
            if idx == self.vocab[word]:
                continue
            
            res_word = self.idx_to_word[idx]
            score = similarities[idx]
            results.append((res_word, score))
            if len(results) >= top_n:
                break
            
        return results

    def solve_analogy(self, positive, negative, top_n=5):
        """
        Solves: Pos1 + Pos2 - Neg1 = ?
        Example: King - Man + Woman = Queen
        """
        # Start with zero vector
        result_vec = np.zeros(self.embeddings.shape[1])
        
        input_words = [w.lower() for w in positive + negative]
        
        # Add positives
        for word in positive:
            v = self.get_vector(word.lower())
            if v is None: return f"Error: '{word}' not in vocab"
            result_vec += v
            
        # Subtract negatives
        for word in negative:
            v = self.get_vector(word.lower())
            if v is None: return f"Error: '{word}' not in vocab"
            result_vec -= v
            
        # Normalize result vector to unit length
        result_norm = np.linalg.norm(result_vec)
        if result_norm != 0:
            result_vec = result_vec / result_norm
        
        # Search
        similarities = np.dot(self.normalized_embeddings, result_vec)
        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in sorted_indices:
            word = self.idx_to_word[idx]
            # Don't return the input words themselves
            if word not in input_words:
                results.append((word, similarities[idx]))
                if len(results) >= top_n: break
                
        return results

    def doesnt_match(self, words):
        """Identifies the word that is least similar to the mean of the others."""
        vectors = []
        valid_words = []
        for w in words:
            v = self.get_vector(w.lower())
            if v is not None:
                vectors.append(v)
                valid_words.append(w)
        
        if len(vectors) < 2:
            return "Need at least 2 valid words"
            
        mean_vec = np.mean(vectors, axis=0)
        
        # Calculate distance of each word to the mean
        similarities = []
        for w, v in zip(valid_words, vectors):
            # We want the one with lowest similarity to mean
            sim = np.dot(v, mean_vec)
            similarities.append((w, sim))
            
        # Sort by similarity ascending (lowest first)
        similarities.sort(key=lambda x: x[1])
        return similarities[0] # (word, score)

    def predict_center(self, context_words, top_n=5):
        """Predicts the center word given a context (CBOW style)."""
        vectors = []
        valid_context = []
        for w in context_words:
            v = self.get_vector(w.lower())
            if v is not None:
                vectors.append(v)
                valid_context.append(w)
                
        if not vectors:
            return "No valid context words"
            
        # Sum vectors (standard CBOW logic)
        context_vec = np.sum(vectors, axis=0)
        
        # Normalize for cosine similarity search
        norm = np.linalg.norm(context_vec)
        if norm > 0:
            context_vec /= norm
            
        # Search
        similarities = np.dot(self.normalized_embeddings, context_vec)
        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in sorted_indices:
            word = self.idx_to_word[idx]
            # Usually we don't exclude context words in prediction, 
            # but sometimes it helps avoid just repeating the context.
            # Let's keep them in to see true model behavior.
            results.append((word, similarities[idx]))
            if len(results) >= top_n: break
            
        return results

    def plot_clusters(self, words_to_plot):
        """
        Reduces dimensions to 2D using t-SNE and plots them.
        """
        print("Calculating 2D projection (t-SNE)...")
        vectors = []
        labels = []
        
        for word in words_to_plot:
            word = word.lower().strip()
            if word in self.vocab:
                vectors.append(self.get_vector(word))
                labels.append(word)
            else:
                print(f"Skipping '{word}' (not in vocab)")
        
        if not vectors:
            print("No valid words to plot.")
            return

        vectors = np.array(vectors)
        
        # Perplexity must be less than number of samples
        perp = min(30, len(vectors)-1)
        if perp < 1: perp = 1
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
        reduced_vectors = tsne.fit_transform(vectors)
        
        plt.figure(figsize=(12, 8))
        plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c='blue', edgecolors='k', alpha=0.7)
        
        for i, label in enumerate(labels):
            plt.annotate(label, (reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=11, alpha=0.9)
            
        plt.title("Word Vector Clusters (WikiText)")
        plt.grid(True, alpha=0.3)
        
        try:
            plt.savefig("plot.png")
            print("Plot saved to 'plot.png'.")
            # Only try to show if we think we have a display
            if os.environ.get('DISPLAY'):
                plt.show()
        except:
            pass

# ==========================================
# RUN LOGIC
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Word2Vec Model")
    parser.add_argument("checkpoint", type=str, help="Path to model file (.safetensors or .pth)", nargs='?')
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if checkpoint is None:
        # Default behavior: try safetensors first, then pth
        if os.path.exists("word2vec_sampled_50m.safetensors"):
            checkpoint = "word2vec_sampled_50m.safetensors"
        elif os.path.exists("word2vec_sampled_50m.pth"):
            checkpoint = "word2vec_sampled_50m.pth"
        else:
            print("Error: No default model found (checked .safetensors and .pth)")
            sys.exit(1)

    if not os.path.exists(checkpoint):
        print(f"Checkpoint file '{checkpoint}' not found.")
        sys.exit(1)

    evaluator = Word2VecEvaluator(checkpoint)
    
    # --- BENCHMARK ---
    print("\n--- QUICK BENCHMARK ---")
    
    # 1. Similarity Check
    test_word = "good"
    print(f"Nearest to '{test_word}':")
    res = evaluator.most_similar(test_word, top_n=5)
    if res:
        for w, s in res: print(f"  {w:<15} {s:.4f}")
    
    # 2. Analogy Check (The "Wiki" Test)
    print("\nAnalogy Test (king - man + woman):")
    res = evaluator.solve_analogy(['king', 'woman'], ['man'])
    if isinstance(res, str):
        print(f"  {res}")
    else:
        for w, s in res: print(f"  {w:<15} {s:.4f}")

    # 3. Odd One Out Check
    print("\nOdd One Out Test (apple, banana, orange, car):")
    res = evaluator.doesnt_match(['apple', 'banana', 'orange', 'car'])
    if isinstance(res, str):
        print(f"  {res}")
    else:
        w, s = res
        print(f"  Outlier: {w} (score: {s:.4f})")

    # --- INTERACTIVE LOOP ---
    print("\n--- INTERACTIVE MODE ---")
    print("Commands:")
    print("  word            -> Nearest neighbors (e.g., 'computer')")
    print("  a - b + c       -> Analogy (e.g., 'paris - france + italy')")
    print("  outlier: a,b,c  -> Find odd one out")
    print("  context: a,b    -> Predict center word from context")
    print("  plot: a,b,c     -> Save t-SNE plot")
    print("  exit            -> Quit")
    
    while True:
        try:
            query = input("\nQuery> ").strip().lower()
        except KeyboardInterrupt:
            break
            
        if query in ['exit', 'quit']:
            break
        if not query:
            continue
            
        if query.startswith("plot:"):
            # Format: plot: apple, banana, car
            words = [w.strip() for w in query.replace("plot:", "").split(",")]
            evaluator.plot_clusters(words)
            continue
        
        if query.startswith("outlier:"):
            # Format: outlier: apple, banana, car
            words = [w.strip() for w in query.replace("outlier:", "").split(",")]
            res = evaluator.doesnt_match(words)
            if isinstance(res, str):
                print(f"  Error: {res}")
            else:
                w, s = res
                print(f"  Outlier: '{w}' (score: {s:.4f})")
            continue

        if query.startswith("context:"):
            # Format: context: united, states
            words = [w.strip() for w in query.replace("context:", "").split(",")]
            res = evaluator.predict_center(words)
            if isinstance(res, str):
                print(f"  Error: {res}")
            else:
                print(f"  Prediction for context {words}:")
                for w, s in res:
                    print(f"    {w:<15} {s:.4f}")
            continue
            
        if "+" in query or "-" in query:
            # Parse logic: "paris - france + italy"
            parts = query.split()
            positive = []
            negative = []
            current_sign = 1 
            
            try:
                for part in parts:
                    if part == "+":
                        current_sign = 1
                    elif part == "-":
                        current_sign = -1
                    else:
                        if current_sign == 1: positive.append(part)
                        else: negative.append(part)
                            
                results = evaluator.solve_analogy(positive, negative)
                if isinstance(results, str):
                    print(results)
                else:
                    print(f"Result: {positive} - {negative}")
                    for w, score in results:
                        print(f"  {w:<15} {score:.4f}")
            except Exception as e:
                print(f"Error parsing analogy: {e}")
            continue

        # Default: Nearest Neighbors
        results = evaluator.most_similar(query)
        if results is None:
            print(f"Word '{query}' not in vocabulary.")
        else:
            print(f"Neighbors for '{query}':")
            for w, score in results:
                print(f"  {w:<15} {score:.4f}")