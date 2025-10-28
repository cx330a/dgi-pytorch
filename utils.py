import os
import random
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def normalize_adjacency(A):
    """A: [N,N] binary adjacency (numpy)
       returns  Ā = D^{-1/2} (A + I) D^{-1/2} (numpy)"""
    A = A.astype(np.float32)
    N = A.shape[0]
    I = np.eye(N, dtype=np.float32)
    A_hat = A + I
    d = A_hat.sum(axis=1)
    d_inv_sqrt = np.power(d, -0.5, where=d>0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return A_norm

def generate_sbm_graph(n_nodes=300, n_communities=3, p_in=0.08, p_out=0.01, feat_dim=128):
    """Generate a small stochastic block model (SBM) graph with Gaussian features.
       Returns (A, X, y) as numpy arrays.
    """
    assert n_nodes % n_communities == 0, "n_nodes must be divisible by n_communities for simplicity."
    nodes_per = n_nodes // n_communities
    # community labels
    y = np.repeat(np.arange(n_communities), nodes_per)
    # adjacency
    A = np.zeros((n_nodes, n_nodes), dtype=np.uint8)
    rng = np.random.default_rng()

    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            same = (y[i] == y[j])
            p = p_in if same else p_out
            if rng.random() < p:
                A[i, j] = 1
                A[j, i] = 1

    # features: community-specific Gaussian means
    X = np.zeros((n_nodes, feat_dim), dtype=np.float32)
    means = np.eye(n_communities, dtype=np.float32)[:,:min(n_communities, feat_dim)]
    means = np.pad(means, ((0,0), (0, feat_dim-means.shape[1])), mode="constant")
    for c in range(n_communities):
        idx = np.where(y == c)[0]
        mu = means[c] * 2.0
        X[idx] = mu + 0.5 * rng.standard_normal(size=(len(idx), feat_dim)).astype(np.float32)

    return A, X, y

def plot_tsne(embeddings, labels, save_path="results/tsne_plot.png"):
    """Create a single-axes t-SNE scatter; do not specify colors to comply with plotting constraints."""
    ensure_dir(os.path.dirname(save_path) if os.path.dirname(save_path) else ".")
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30)
    Z = tsne.fit_transform(embeddings)
    plt.figure(figsize=(6,6))
    plt.scatter(Z[:,0], Z[:,1], s=6, c=labels)
    plt.title("t-SNE of DGI Embeddings")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
