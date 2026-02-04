import torch
import numpy as np
import argparse
import os
import sys
import warnings

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
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # 1. Extract Vocab
            self.vocab = checkpoint['vocab']
            self.idx_to_word = checkpoint['idx_to_word']
            
            # 2. Extract Weights
            # We need to handle both 'Legacy' and 'Modern' architectures
            state_dict = checkpoint['model_state_dict']
            
            if 'in_embeddings.weight' in state_dict:
                # Modern Architecture (Negative Sampling)
                self.embeddings = state_dict['in_embeddings.weight'].numpy()
            elif 'embeddings.weight' in state_dict:
                # Legacy Architecture (Standard CBOW)
                self.embeddings = state_dict['embeddings.weight'].numpy()
            else:
                print("Error: Could not find embeddings in checkpoint.")
                print(f"Available keys: {state_dict.keys()}")
                sys.exit(1)
                
            print(f"Model Loaded. Vocab Size: {len(self.vocab)}, Dimensions: {self.embeddings.shape[1]}")
            
            # 3. Pre-normalize vectors for fast Cosine Similarity
            # Formula: (A . B) / (|A| * |B|)
            # If we normalize A and B to length 1 first, it becomes just (A . B)
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

def run_suite(evaluator):
    print("\n" + "="*40)
    print("      AUTOMATED WORD2VEC TEST SUITE")
    print("="*40)

    # 1. Similarity Tests
    sim_tests = [
        "good", "bad", "happy", "sad",
        "computer", "science", "physics",
        "king", "queen", "man", "woman",
        "apple", "banana", "fruit",
        "car", "train", "plane",
        "dog", "cat", "animal",
        "red", "blue", "green"
    ]
    
    print("\n[ TEST SECTION 1: Nearest Neighbors ]")
    for word in sim_tests:
        print(f"\n--> Neighbors for '{word}':")
        res = evaluator.most_similar(word, top_n=5)
        if res:
            for w, s in res:
                print(f"    {w:<15} {s:.4f}")
        else:
            print(f"    (Word not in vocabulary)")

    # 2. Analogy Tests
    # Format: (positive, negative, expected_hint)
    # Target = Pos1 + Pos2 - Neg1
    analogy_tests = [
        (['king', 'woman'], ['man'], "queen"),
        (['paris', 'germany'], ['france'], "berlin"),
        (['london', 'france'], ['england'], "paris"),
        (['walk', 'swimming'], ['walking'], "swim"),
        (['good', 'worse'], ['bad'], "better"),
        (['man', 'actress'], ['woman'], "actor"),
        (['brother', 'woman'], ['man'], "sister"),
        (['japan', 'vodka'], ['russia'], "sake"),
        (['usa', 'pizza'], ['italy'], "burger"),
        (['cars', 'one'], ['car'], "ones"), # Plural grammar check
        (['write', 'did'], ['do'], "wrote"), # Past tense check
    ]

    print("\n[ TEST SECTION 2: Analogies ]")
    print("Formula: Pos1 + Pos2 - Neg1 = ?")
    
    for pos, neg, hint in analogy_tests:
        print(f"\n--> {pos} - {neg} (Expect: ~{hint})")
        res = evaluator.solve_analogy(pos, neg, top_n=5)
        if isinstance(res, str):
            print(f"    Error: {res}")
        else:
            for i, (w, s) in enumerate(res):
                marker = " <--" if hint.lower() in w.lower() else ""
                print(f"    {i+1}. {w:<15} {s:.4f}{marker}")

    # 3. Odd One Out Tests
    odd_one_out_tests = [
        ["apple", "banana", "orange", "car"],      # car
        ["red", "blue", "green", "dog"],           # dog
        ["computer", "science", "physics", "cow"], # cow
        ["king", "queen", "prince", "toaster"],    # toaster
        ["germany", "france", "italy", "tokyo"],   # tokyo (city vs countries)
    ]
    
    print("\n[ TEST SECTION 3: Odd One Out ]")
    for words in odd_one_out_tests:
        res = evaluator.doesnt_match(words)
        if isinstance(res, str):
            print(f"  Error: {res}")
        else:
            w, score = res
            print(f"  {words} -> '{w}' (score: {score:.4f})")

    # 4. Fill in the Middle (Context Prediction)
    print("\n[ TEST SECTION 4: Fill in the Middle (Context Prediction) ]")
    context_tests = [
        (["machine", "learning"], "intelligence"),
        (["united", "states"], "america"),
        (["new", "york"], "city"),
        (["social", "media"], "network"),
        (["artificial", "intelligence"], "machine"),
        (["deep", "learning"], "neural"),
    ]
    
    for context, hint in context_tests:
        print(f"\n--> Context: {context} (Expect: ~{hint})")
        res = evaluator.predict_center(context, top_n=5)
        if isinstance(res, str):
            print(f"    Error: {res}")
        else:
            for i, (w, s) in enumerate(res):
                marker = " <--" if hint.lower() in w.lower() else ""
                print(f"    {i+1}. {w:<15} {s:.4f}{marker}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Word2Vec Tests")
    parser.add_argument("checkpoint", type=str, help="Path to .pth file", nargs='?', default="word2vec_sampled_50m.pth")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint file '{args.checkpoint}' not found.")
        print("Usage: python automated_tests.py <path_to_checkpoint.pth>")
        sys.exit(1)

    evaluator = Word2VecEvaluator(args.checkpoint)
    run_suite(evaluator)
