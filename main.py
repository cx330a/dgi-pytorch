import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from utils import (
    set_seed, get_device, generate_sbm_graph, normalize_adjacency,
    plot_tsne, ensure_dir
)
from model import DGI, GCNEncoder

def train_dgi(x, adj, hidden_dim=128, lr=1e-3, epochs=100, device="cpu"):
    x = x.to(device)
    adj = adj.to(device)

    enc = GCNEncoder(in_dim=x.size(1), out_dim=hidden_dim).to(device)
    model = DGI(encoder=enc, out_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(1, epochs + 1):
        h, h_tilde, s = model(x, adj)
        loss = model.loss(h, h_tilde, s)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % max(1, epochs // 20) == 0:
            print(f"Epoch [{ep}/{epochs}] | Loss: {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        z = model.embed(x, adj).cpu().numpy()
    return z

def main():
    parser = argparse.ArgumentParser(description="Pure-PyTorch Deep Graph Infomax (DGI)")
    parser.add_argument("--nodes", type=int, default=300, help="number of nodes")
    parser.add_argument("--communities", type=int, default=3, help="number of communities")
    parser.add_argument("--p_in", type=float, default=0.08, help="intra-community edge prob")
    parser.add_argument("--p_out", type=float, default=0.01, help="inter-community edge prob")
    parser.add_argument("--feat_dim", type=int, default=128, help="input feature dim")
    parser.add_argument("--hidden_dim", type=int, default=128, help="hidden dim")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Generating random SBM graph with {args.nodes} nodes, {args.communities} communities...")
    A_np, X_np, y_np = generate_sbm_graph(
        n_nodes=args.nodes,
        n_communities=args.communities,
        p_in=args.p_in,
        p_out=args.p_out,
        feat_dim=args.feat_dim
    )
    # Normalize adjacency once
    A_norm = normalize_adjacency(A_np)

    # Torch tensors
    x = torch.from_numpy(X_np).float()
    adj = torch.from_numpy(A_norm).float()

    print(f"[INFO] Training DGI ({args.epochs} epochs) ...")
    z = train_dgi(
        x=x,
        adj=adj,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        epochs=args.epochs,
        device=device
    )

    ensure_dir("results")
    np.save("results/embeddings.npy", z)
    np.save("results/labels.npy", y_np)
    print("[INFO] Saved embeddings to results/embeddings.npy")

    print("[INFO] Plotting t-SNE visualization ...")
    plot_tsne(z, y_np, save_path="results/tsne_plot.png")
    print("✅ DGI training complete. See results/ for outputs.")

if __name__ == "__main__":
    main()
