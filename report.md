# Deep Graph Infomax (DGI) — Technical Report (Concise)

## Problem
Learn node embeddings on a graph **without labels** that are useful for downstream tasks such as node classification and clustering.

## Idea
Maximize **mutual information** between **local** node representations \(h_i\) and a **global** summary \(s\) of the entire graph. This encourages each node representation to contain global statistics of the input graph.

## Architecture
1. **Encoder** \(f_\theta\): GCN mapping \((X, A)\) to embeddings \(H=\{h_i\}\).
2. **Readout** \(R(H)\): global summary \(s=\sigma(\text{mean}(H))\).
3. **Discriminator** \(D(h, s)=\sigma(h^\top W s)\): distinguishes positive pairs \((h_i, s)\) from negative pairs \((\tilde{h}_i, s)\) where \(\tilde{h}\) is computed on a **corrupted** view (feature permutation).

## Objective
Binary cross-entropy for positives vs. negatives (a JSD lower bound on MI):
\[
\mathcal{L} = - \mathbb{E}_i[\log D(h_i, s)] - \mathbb{E}_j[\log (1 - D(\tilde{h}_j, s))].
\]

## Why It Works
If \(h_i\) contains global information useful to predict \(s\), positives become distinguishable from corrupted negatives. This effectively **maximizes** a lower bound on the **mutual information** \(I(H; s)\).

## This Repo
- Implements a shallow **GCN** and the **DGI** objective in **pure PyTorch**.
- Uses a synthetic **SBM graph** to emulate community structures.
- Trains DGI and visualizes embeddings using **t-SNE**.

## Reproduction Steps
1. Generate SBM graph (adjacency, features, labels).
2. Normalize adjacency \(\hat{A}=D^{-1/2}(A+I)D^{-1/2}\).
3. Train DGI for N epochs with corruption = row permutation of features.
4. Extract embeddings and visualize with t-SNE.

## Practical Tips
- Shallow encoders often perform best for DGI on homophilic graphs.
- Keep corruption simple (feature permutation) for stability.
- Tune hidden dim (64–512) and LR (1e-3 ~ 5e-4).

## Outputs
- `results/embeddings.npy`, `results/labels.npy`, `results/tsne_plot.png`.
