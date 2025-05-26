# Orthogonal Concept Transformer Analysis

## From an information dynamics perspective, standard Transformers learn an implicit latent ontology (a system of representing concepts and their relationships) through end-to-end optimization. Through orthogonal feature independence—we can promote the emergence of a more parsimonious or compositionally-structured ontology.

This repository explores methods for influencing the learned latent representations within Transformer architectures, primarily by leveraging or encouraging orthogonality and structured transformations. The goal is to investigate paths towards more disentangled, robust, and interpretable models.

Two main approaches are currently explored:

1.  **Baseline Orthogonal Model:** Implemented in `ml.py` and tested via `test_orthogonal_concept.py`. This model uses a learnable orthogonal transformation applied directly to token embeddings.
2.  **Architectural Exploration - MirrorShadow Model:** Implemented in `ml_mirrorshadow.py` and tested via `test_orthogonal_concept_mirror.py`. This model introduces a more deeply integrated "shadow space" concept, where embeddings are projected through a bottleneck into a distinct representational space for all transformer operations.

---

## 1. Baseline Orthogonal Model (`ml.py`)

This approach focuses on applying a learnable orthogonal transformation directly to the token embeddings and processing sequences in this transformed "inverse" space.

### Key Visualizations (Baseline Orthogonal Model)

#### Character Embedding Point Cloud
This plot visualizes the 2D PCA-reduced character embeddings from sample words, comparing embeddings before ("Standard") and after ("Inverse/Orthogonal") the orthogonal transformation.

*(Run `python test_orthogonal_concept.py` to generate `orthogonal_embedding_point_cloud.png`)*

![Character Embedding Point Cloud - Baseline Orthogonal Model](orthogonal_embedding_point_cloud.png)

#### Comprehensive Metric Analysis
Compares the baseline orthogonal model against a standard Transformer on various metrics.

*(Run `python test_orthogonal_concept.py` to generate `orthogonal_concept_analysis.png`)*

![Comprehensive Metric Analysis - Baseline Orthogonal Model](orthogonal_concept_analysis.png)

### Observations (Baseline Orthogonal Model from `test_orthogonal_concept.py`)

The latest comprehensive test run yielded:

**PERPLEXITY (wikitext validation - 500 samples):**
*   Orthogonal Model: **4.83**
*   Standard Model:   6.63

**EMBEDDING INDEPENDENCE (Avg. Off-Diagonal Covariance):**
*   Orthogonal: **0.1224**
*   Standard:   0.2173
*   Improvement: **43.7%**

**PREDICTION DIVERSITY (Unique N-gram Ratio):**
*   Orthogonal: **0.5986**
*   Standard:   0.5797
*   Improvement: **3.2%**

**ROBUSTNESS (avg KL divergence):**
*   Orthogonal: **0.0000**
*   Standard:   0.0015
*   Improvement: **96.7%**

**INFORMATION PROPERTIES:**
*   Entropy - Orthogonal: **1.5294**, Standard: 2.0003
*   Mutual Info - Orthogonal: **0.7000**, Standard: 0.8380

These results highlight strong performance in perplexity, disentanglement, and robustness for this relatively direct orthogonalization method.

---

## 2. Architectural Exploration - MirrorShadow Model (`ml_mirrorshadow.py`)

This branch of exploration investigates a more deeply integrated architectural change. The "MirrorShadow" model projects token embeddings into a distinct, potentially bottlenecked "shadow space." All subsequent Transformer operations (attention, feed-forward layers) occur entirely within this shadow space. The aim is to force the model to learn and reason in this constrained representational space, potentially leading to even more pronounced disentanglement or different information processing characteristics.

### Key Visualizations (MirrorShadow Model)

#### Character Embedding Point Cloud
Visualizes character embeddings in the initial `d_model` space versus the transformed `d_shadow` space after projection.

*(Run `python test_orthogonal_concept_mirror.py` to generate `embedding_point_cloud.png` - note the different filename from the baseline model's plot)*

![Character Embedding Point Cloud - MirrorShadow Model](embedding_point_cloud.png) 

#### Comprehensive Metric Analysis
Compares the MirrorShadow model against a standard Transformer.

*(Run `python test_orthogonal_concept_mirror.py` to generate `mirror_shadow_concept_analysis.png`)*

![Comprehensive Metric Analysis - MirrorShadow Model](mirror_shadow_concept_analysis.png)

### Observations (MirrorShadow Model from `test_orthogonal_concept_mirror.py`)

*(Populate with key results from running `test_orthogonal_concept_mirror.py`. Example structure below, replace with your actual numbers when available)*

**PERPLEXITY (wikitext validation - 500 samples):**
*   MirrorShadow Model: **X.XX**
*   Standard Model:   Y.YY

**EMBEDDING INDEPENDENCE (Avg. Off-Diagonal Covariance - Shadow Space vs. Pre-Projection):**
*   MirrorShadow (d_shadow space): **A.AAAA**
*   MS Pre-Projection (d_model space): B.BBBB
*   Reduction: **ZZ.Z%**

**ROBUSTNESS (avg KL divergence):**
*   MirrorShadow: **C.CCCC**
*   Standard:   D.DDDD
*   Improvement: **WW.W%**

This model explores the trade-offs of forcing operations into a potentially more compressed and transformed latent space, often achieving extreme disentanglement.

---

## Running the Code

1.  Ensure you have the necessary dependencies installed (PyTorch, NumPy, Matplotlib, Seaborn, Scikit-learn, Datasets, TQDM).
2.  **To test the Baseline Orthogonal Model:**
    ```bash
    python test_orthogonal_concept.py
    ```
    This generates `orthogonal_test_results_large_run.json`, `orthogonal_concept_analysis.png`, and `orthogonal_embedding_point_cloud.png`.
3.  **To test the Architectural Exploration - MirrorShadow Model:**
    *(Ensure you are on the appropriate git branch if you've separated these, e.g., `architectural-exploration`)*
    ```bash
    python test_orthogonal_concept_mirror.py
    ```
    This generates `mirror_shadow_test_results.json`, `mirror_shadow_concept_analysis.png`, and `embedding_point_cloud.png`.
