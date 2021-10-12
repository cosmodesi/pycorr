"""Package wrapping different pair counting codes, handling PIP and angular upweights."""

HAS_MPI = True
try:
    from . import mpi
except ImportError:
    HAS_MPI = False

from .pair_counter import TwoPointCounter, AnalyticTwoPointCounter, BaseTwoPointCounter
from .estimator import TwoPointEstimator, NaturalTwoPointEstimator, LandySzalayTwoPointEstimator, project_to_multipoles
from .correlation_function import TwoPointCorrelationFunction
from .utils import setup_logging
