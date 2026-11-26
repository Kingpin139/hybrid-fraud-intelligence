"""Data utilities (synthetic toy data; real loaders are on the roadmap)."""

from .synthetic import SyntheticFraudGraph, generate_synthetic_fraud_graph, stratified_node_split

__all__ = ["SyntheticFraudGraph", "generate_synthetic_fraud_graph", "stratified_node_split"]
