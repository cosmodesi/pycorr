"""Package wrapping different pair counting codes, handling PIP and angular upweights."""

HAS_MPI = True
try:
    from . import mpi
except ImportError:
    HAS_MPI = False

from .pair_counter import TwoPointCounter, AnalyticTwoPointCounter
from .estimator import TwoPointEstimator
from .correlation_function import TwoPointCorrelationFunction
