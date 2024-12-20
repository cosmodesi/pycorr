"""Package wrapping different pair counting codes, handling PIP and angular upweights."""

from ._version import __version__
from .twopoint_counter import TwoPointCounter, AnalyticTwoPointCounter
from .twopoint_jackknife import BoxSubsampler, KMeansSubsampler, JackknifeTwoPointCounter, JackknifeTwoPointEstimator
from .twopoint_estimator import TwoPointEstimator, NaturalTwoPointEstimator, LandySzalayTwoPointEstimator,\
                                project_to_poles, project_to_wedges, project_to_wp
from .twopoint_estimator import project_to_multipoles  # for backward-compatibility
from .correlation_function import TwoPointCorrelationFunction
from .utils import setup_logging
