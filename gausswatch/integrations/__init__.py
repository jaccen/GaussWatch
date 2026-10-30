"""Framework integrations (torch-side adapters).

These modules import torch lazily: the core gausswatch package stays
zero-dependency, and only code paths that touch training tensors require it.
"""
