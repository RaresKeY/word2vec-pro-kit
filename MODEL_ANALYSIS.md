# Word2Vec Model Comparison Analysis

This report compares the performance and efficiency of the model in three states: Full (Raw), Pruned (FP32), and Pruned & Quantized (FP16).

## 1. Efficiency Metrics

| Model Variant | Precision | File Size | Reduction % | Training Artifacts |
| :--- | :--- | :--- | :--- | :--- |
| **Full** | FP32 | 229 MB | 0% | Included (`out_embeddings`) |
| **Pruned** | FP32 | 115 MB | ~50% | Removed |
| **Pruned + FP16** | FP16 | 58 MB | **~75%** | Removed |

## 2. Semantic Performance Comparison

Analysis of the `automated_tests.py` suite across all three models:

### Nearest Neighbors (`good`)
*   **Full (FP32):** `bad` (0.6594), `decent` (0.4800)
*   **Pruned (FP32):** `bad` (0.6594), `decent` (0.4800)
*   **Pruned (FP16):** `bad` (0.6597), `decent` (0.4800)
*   **Verdict:** Identical rankings. Score variance < 0.05%.

### Analogies (`king - man + woman`)
*   **Full (FP32):** `queen` (0.5649)
*   **Pruned (FP32):** `queen` (0.5649)
*   **Pruned (FP16):** `queen` (0.5647)
*   **Verdict:** All models successfully solved the analogy with virtually identical confidence.

### Odd One Out (`apple, banana, orange, car`)
*   **Full (FP32):** `car` (0.3463)
*   **Pruned (FP32):** `car` (0.3463)
*   **Pruned (FP16):** `car` (0.3462)
*   **Verdict:** Identical logic across all precision levels.

## 3. Engineering Conclusion

1.  **Pruning is mandatory:** The `out_embeddings` weights are only useful for updating gradients during training. They provide zero value for inference (similarity/analogies) and double the file size.
2.  **FP16 is the "Sweet Spot":** Converting to Half-Precision (FP16) reduces the model size by another 50% (58MB total) with no observable degradation in the quality of results.
3.  **Deployment Recommendation:** Use the **Pruned + FP16** variant for all production and sharing scenarios. It offers the best balance of speed, storage, and accuracy.
