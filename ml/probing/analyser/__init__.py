from .pca import PCAAnalysis
from .knn import KNNAnalysis
from .linear import LinearAnalysis
from .logreg import LogRegAnalysis
from .lda import LinDiscrAnalysis

__all__ = [
    "LinearAnalysis",
    "KNNAnalysis",
    "LogRegAnalysis",
    "LinDiscrAnalysis",
    "PCAAnalysis",
]
