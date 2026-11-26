"""Example inference script with a small synthetic graph."""

import torch
from torch_geometric.data import HeteroData

from hybrid_fraud.models import HybridFraudModel

NUM_USERS = 6
NUM_MERCHANTS = 3
NUM_DEVICES = 4
FEAT = 32


def create_dummy_batch(batch_size: int = 4, hidden: int = 64):
    """
    Create a minimal valid HeteroData plus per-target temporal/static inputs.

    The graph holds NUM_USERS users, but only ``batch_size`` of them are scored;
    ``user_index`` says which ones.
    """
    data = HeteroData()
    data["user"].x = torch.randn(NUM_USERS, FEAT)
    data["merchant"].x = torch.randn(NUM_MERCHANTS, FEAT)
    data["device"].x = torch.randn(NUM_DEVICES, FEAT)

    # (src indices < #users, dst indices < #dst nodes)
    def edges(n_src: int, n_dst: int, n_edges: int) -> torch.Tensor:
        return torch.stack(
            [torch.randint(0, n_src, (n_edges,)), torch.randint(0, n_dst, (n_edges,))]
        )

    data["user", "transacts", "merchant"].edge_index = edges(NUM_USERS, NUM_MERCHANTS, 20)
    data["user", "uses", "device"].edge_index = edges(NUM_USERS, NUM_DEVICES, 15)
    # Reverse edges so messages flow back to the users
    data["merchant", "rev_transacts", "user"].edge_index = data[
        "user", "transacts", "merchant"
    ].edge_index.flip(0)
    data["device", "rev_uses", "user"].edge_index = data["user", "uses", "device"].edge_index.flip(
        0
    )

    user_index = torch.arange(batch_size)  # score the first `batch_size` users
    temporal = torch.randn(batch_size, 8, hidden)
    static = torch.randn(batch_size, FEAT)
    return data, temporal, static, user_index


def main():
    print("Hybrid Fraud Intelligence — Inference Demo")
    print("=" * 50)

    model = HybridFraudModel(static_dim=FEAT, hidden_channels=64, num_gnn_layers=2)
    model.eval()

    graph, temporal, static, user_index = create_dummy_batch()

    with torch.no_grad():
        logits, weights = model(graph, temporal, static, user_index=user_index)
        probs = torch.softmax(logits, dim=-1)
        risk = probs[:, 1]  # fraud class probability

    print(
        f"Fusion weights: α={weights['alpha']:.3f}, "
        f"β={weights['beta']:.3f}, γ={weights['gamma']:.3f}"
    )
    print(f"Sample risk scores (untrained model, random data): {risk[:3].tolist()}")
    print("\nDemo completed successfully.")


if __name__ == "__main__":
    main()
