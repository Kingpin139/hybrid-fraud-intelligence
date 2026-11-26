import pytest
import torch
from torch_geometric.data import HeteroData


def make_edges(n_src: int, n_dst: int, n_edges: int) -> torch.Tensor:
    """Random edges with src/dst indices drawn from their own node-type ranges."""
    return torch.stack([torch.randint(0, n_src, (n_edges,)), torch.randint(0, n_dst, (n_edges,))])


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def graph() -> HeteroData:
    """5 users, 3 merchants, 4 phones; forward + reverse edges."""
    data = HeteroData()
    data["user"].x = torch.randn(5, 16)
    data["merchant"].x = torch.randn(3, 16)
    data["phone"].x = torch.randn(4, 8)  # different feature size on purpose
    data["user", "transacts", "merchant"].edge_index = make_edges(5, 3, 10)
    data["merchant", "rev_transacts", "user"].edge_index = data[
        "user", "transacts", "merchant"
    ].edge_index.flip(0)
    data["user", "shares", "phone"].edge_index = make_edges(5, 4, 6)
    data["phone", "rev_shares", "user"].edge_index = data[
        "user", "shares", "phone"
    ].edge_index.flip(0)
    return data
