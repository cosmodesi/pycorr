import numpy as np

from .estimator import get_estimator
from .pair_counter import TwoPointCounter, AnalyticTwoPointCounter
from .utils import BaseClass


def TwoPointCorrelationFunction(mode, edges, data_positions1, data_positions2=None, randoms_positions1=None, randoms_positions2=None,
                                data_weights1=None, data_weights2=None, randoms_weights1=None, randoms_weights2=None,
                                estimator=None, boxsize=None, **kwargs):


        has_randoms = randoms_positions1 is not None
        Estimator = get_estimator(estimator, has_randoms=has_randoms)

        autocorr = data_positions2 is None or (data_positions2 is data_positions1 and data_weights2 is data_weights1)

        if autocorr:
            data_positions2 = data_positions1
            data_weights2 = data_weights1
            randoms_positions2 = randoms_positions1
            randoms_weights2 = randoms_weights1

        positions = {'D1':data_positions1, 'D2':data_positions2, 'R1':randoms_positions1, 'R2':randoms_positions2}
        weights = {'D1':data_weights1, 'D2':data_weights2, 'R1':randoms_weights1, 'R2':randoms_weights2}

        pairs = {}
        for label1,label2 in Estimator.requires(autocorr=True):
            if label1+label2 == 'R1R2' and not has_randoms:
                pairs[label1+label2] = AnalyticTwoPointCounter(mode, edges, boxsize,
                                                               n1=positions[label1][0].size, positions2=positions[label2][0].size)

            pairs[label1+label2] = TwoPointCounter(mode, edges, positions[label1], positions2=positions[label2],
                                                   weights1=weights[label1], weights2=weights[label2], **kwargs)

        return Estimator(**pairs)
