# Orthogonal Concept Transformer Analysis

## From an information dynamics perspective, standard Transformers learn an implicit latent ontology (a system of representing concepts and their relationships) through end-to-end optimization. Through orthogonal feature independence—we can promote the emergence of a more parsimonious or compositionally-structured ontology.

This repository explores the concept of using orthogonal transformations within a Transformer architecture to influence the learned latent representations. The primary goal is to investigate whether enforcing orthogonality can lead to more disentangled, robust, and interpretable models.

The core idea is implemented in `ml.py` and tested via `test_orthogonal_concept.py`.

## Key Visualizations

### Character Embedding Point Cloud (Orthogonal Model)

This plot visualizes the 2D PCA-reduced character embeddings from a set of sample words. It compares the embeddings before the orthogonal transformation ("Standard Embeddings") and after the transformation ("Inverse/Orthogonal Embeddings"). This helps to qualitatively assess how the orthogonalization reshapes the representational space.

*(You will need to run `test_orthogonal_concept.py` to generate this image. It will be saved as `orthogonal_embedding_point_cloud.png`)*

![Character Embedding Point Cloud](orthogonal_embedding_point_cloud.png)

### Comprehensive Metric Analysis

The `test_orthogonal_concept.py` script also generates a comprehensive analysis of various metrics, comparing the orthogonal model against a standard Transformer baseline. These metrics include perplexity, embedding independence, prediction diversity, robustness to perturbation, and information-theoretic properties (entropy, mutual information).

*(This plot is saved as `orthogonal_concept_analysis.png` when `test_orthogonal_concept.py` is run.)*

![Comprehensive Metric Analysis](orthogonal_concept_analysis.png)

## Observations from `test_orthogonal_concept.py`

The latest comprehensive test run yielded the following key performance metrics:

**PERPLEXITY (on wikitext validation - 500 samples):**
*   Orthogonal Model: **4.83**
*   Standard Model:   6.63

**1. EMBEDDING INDEPENDENCE (Avg. Off-Diagonal Covariance):**
*   Orthogonal: **0.1224**
*   Standard:   0.2173
*   Improvement: **43.7%**

**2. PREDICTION DIVERSITY (Unique N-gram Ratio):**
*   Orthogonal: **0.5986**
*   Standard:   0.5797
*   Improvement: **3.2%**

**3. ROBUSTNESS (avg KL divergence from perturbed inputs):**
*   Orthogonal: **0.0000** 
*   Standard:   0.0015
*   Improvement (lower KL is better): **96.7%**

**4. INFORMATION PROPERTIES:**
*   Entropy (Output Prediction) - Orthogonal: **1.5294**, Standard: 2.0003
*   Mutual Info (Input vs. Hidden Repr.) - Orthogonal: **0.7000**, Standard: 0.8380

These results highlight that the orthogonal model:
- Achieves significantly better perplexity than the standard model.
- Demonstrates substantially improved embedding independence, indicating more disentangled features.
- Shows a slight improvement in prediction diversity.
- Exhibits exceptional robustness to input perturbations.
- Operates with lower output prediction entropy (more confident predictions) and lower mutual information between input and its (inverse) hidden representations, suggesting a more compressed or focused encoding of information.

These characteristics strongly suggest that the orthogonalization constraint encourages the model to learn a more structured, disentangled, and efficient latent space, leading to tangible performance benefits.

## Running the Code

1.  Ensure you have the necessary dependencies installed (PyTorch, NumPy, Matplotlib, Seaborn, Scikit-learn, Datasets, TQDM).
2.  Execute the main test script:
    ```bash
    python test_orthogonal_concept.py
    ```
3.  The script will train both the orthogonal model and a standard baseline, run a series of tests, save detailed results to `orthogonal_test_results_large_run.json`, and generate the `orthogonal_concept_analysis.png` and `orthogonal_embedding_point_cloud.png` visualizations.
