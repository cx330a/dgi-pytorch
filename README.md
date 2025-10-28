# Deep Graph Infomax (DGI) — Pure PyTorch Demo

This is a **pure PyTorch** (no PyTorch Geometric) reproduction of **Deep Graph Infomax (DGI)** on a small **synthetic stochastic block model (SBM)** graph.  
It automatically detects GPU (CUDA) if available.

## Features
- Lightweight **GCN** layer implemented from scratch
- **DGI** objective with corruption by feature permutation
- **t-SNE** visualization of learned embeddings
- Single command to run

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
python main.py --nodes 300 --communities 3 --epochs 100
```
Results will be saved in `results/`:
- `embeddings.npy`
- `labels.npy`
- `tsne_plot.png`

## Arguments
- `--nodes` (int): number of nodes (default: 300)
- `--communities` (int): number of communities (default: 3)
- `--p_in` (float): intra-community edge probability
- `--p_out` (float): inter-community edge probability
- `--feat_dim` (int): input feature dimension (default: 128)
- `--hidden_dim` (int): GCN hidden/embedding dimension (default: 128)
- `--epochs` (int): training epochs (default: 100)
- `--lr` (float): learning rate (default: 1e-3)
- `--seed` (int): random seed (default: 0)

## Notes
- This demo focuses on clarity and portability. For large graphs and benchmark datasets (Cora/Citeseer/Pubmed/Reddit/PPI), consider using a library such as PyG.
