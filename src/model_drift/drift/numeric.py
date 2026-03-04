#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.special import kolmogi
from scipy.stats import ks_2samp, wasserstein_distance
import ot
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance

from model_drift.drift.base import BaseDriftCalculator

class NumericBaseDriftCalculator(BaseDriftCalculator):
    def convert(self, arg):
        return pd.to_numeric(arg, errors="coerce")


class KSDriftCalculator(NumericBaseDriftCalculator):
    name = "ks"

    def __init__(self, q_val=0.1, alternative='two-sided', mode='asymp', average='macro', include_critical_value=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.q_val = q_val
        self.alternative = alternative
        self.mode = mode
        self.average = average
        self.include_critical_value = include_critical_value

    def _predict(self, sample):
        nref = len(self._ref)
        nobs = len(sample)
        out = {}
        try:
            out["distance"], out['pval'] = ks_2samp(self._ref, sample, alternative=self.alternative,
                                                    mode=self.mode)
        except TypeError:
            out["distance"], out['pval'] = float("NaN"), float("NaN")

        if self.include_critical_value:
            out['critical_value'] = self.calc_critical_value(nref, nobs, self.q_val)
            out['critical_diff'] = out["distance"] - out['critical_value']

        return out

    @staticmethod
    def calc_critical_value(n1, n2, q=.01):
        return kolmogi(q) * np.sqrt((n1 + n2) / (n1 * n2))


class KSDriftCalculatorJackKnife(NumericBaseDriftCalculator):
    name = "ks_jackknife"

    def __init__(self, q_val=0.1, alternative='two-sided', mode='asymp', average='macro', include_critical_value=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.q_val = q_val
        self.alternative = alternative
        self.mode = mode
        self.average = average
        self.include_critical_value = include_critical_value

    def _predict(self, sample):
        nref = len(self._ref)
        nobs = len(sample)

        ref1 = np.random.choice(self._ref, nobs)
        ref2 = np.random.choice(self._ref, nobs)
        out = {}
        try:
            dist1, _ = ks_2samp(ref1, sample, alternative=self.alternative, mode=self.mode)
            dist2, _  = ks_2samp(ref1, ref2, alternative=self.alternative, mode=self.mode)

            out["distance"] = max(dist1 - dist2, 0.0)
            out['pval'] = float("NaN")

        except TypeError:
            out["distance"], out['pval'] = float("NaN"), float("NaN")

        if self.include_critical_value:
            raise NotImplementedError("Critical value not implemented for jackknife")
 
        return out


class EMDDriftCalculatorJackKnife_woRef(NumericBaseDriftCalculator):
    name = "emd_jackknife"

    def __init__(self, include_critical_value=False, **kwargs):
        super().__init__(**kwargs)
        self.include_critical_value = include_critical_value
     
    def convert(self, arg):
        return arg

    def _predict(self, sample):

        def _emd_distance(tar, ref):
            a = np.ones((len(ref))) / len(ref)  # Uniform weights for the reference set
            b = np.ones((len(tar))) / len(tar)  # Uniform weights for the target set
            M = ot.dist(ref, tar)
            G0 = ot.emd(a, b, M, numItermax=1000000)
            em_distance = np.sum(M * G0)
            return em_distance
        
        nref = len(self._ref)
        nobs = len(sample)

        sample_tuples = sample.apply(tuple)
        ref_tuples = self._ref.apply(tuple)
        ref_tuples_exclusive = ref_tuples[~ref_tuples.isin(sample_tuples)]
        ref_lists_exclusive = ref_tuples_exclusive.apply(list)

        ref1 = np.random.choice(ref_lists_exclusive, nobs)
        ref2 = np.random.choice(ref_lists_exclusive, nobs)

        sample_arr =  np.array(sample.tolist())
        sample_arr = np.nan_to_num(sample_arr, nan=0.0, posinf=0.0, neginf=0.0)

        ref1_arr =  np.array(ref1.tolist())
        ref1_arr = np.nan_to_num(ref1_arr, nan=0.0, posinf=0.0, neginf=0.0)

        ref2_arr =  np.array(ref2.tolist())
        ref2_arr = np.nan_to_num(ref2_arr, nan=0.0, posinf=0.0, neginf=0.0)

        out = {}
        try:
            dist1  = _emd_distance(ref1_arr, sample_arr)
            dist2  = _emd_distance(ref1_arr, ref2_arr)

            out["distance"] = max(dist1 - dist2, 0.0)
            out['pval'] = float("NaN")

        except TypeError:
            out["distance"], out['pval'] = float("NaN"), float("NaN")

        if self.include_critical_value:
            raise NotImplementedError("Critical value not implemented for jackknife")
 
        return out

class EMDDriftCalculatorJackKnife_1D(NumericBaseDriftCalculator):
    name = "emd_jackknife_1d"

    def __init__(self, include_critical_value=False, **kwargs):
        super().__init__(**kwargs)
        self.include_critical_value = include_critical_value

    def _predict(self, sample):
        nref = len(self._ref)
        nobs = len(sample)

        ref1 = np.random.choice(self._ref, nobs)
        ref2 = np.random.choice(self._ref, nobs)
        out = {}
        # drop NaNs from the arrays before computing the EMD
        ref1 = ref1[~np.isnan(ref1)]
        ref2 = ref2[~np.isnan(ref2)]
        sample = sample[~np.isnan(sample)]
        try:
            dist1 = wasserstein_distance(ref1, sample)
            dist2  = wasserstein_distance(ref1, ref2)

            out["distance"] = max(dist1 - dist2, 0.0)
            out['pval'] = float("NaN")

        except TypeError:
            out["distance"], out['pval'] = float("NaN"), float("NaN")

        if self.include_critical_value:
            raise NotImplementedError("Critical value not implemented for jackknife")
 
        return out

#def permutation_test_mat(matrix, n_1, n_2, n_permutations, a00=1, a11=1, a01=0):
#    """Compute the p-value of the following statistic (rejects when high)
#
#        \sum_{i,j} a_{\pi(i), \pi(j)} matrix[i, j].
#    """
#    if np.isnan(matrix).all():
#        return np.nan
#    n = n_1 + n_2
#    pi = np.zeros(n, dtype=int)
#    pi[n_1:] = 1
#    larger = 0.0
#
#    # Pre-compute the symmetric version of the matrix to avoid redundant calculations
#    symmetric_matrix = matrix + matrix.T
#
#    # Compute initial statistic
#    mij = symmetric_matrix * (pi[:, None] == pi) * a00
#    mij += symmetric_matrix * (pi[:, None] != pi) * a01
#    statistic = np.sum(mij) / 2
#
#    # Main loop over permutations
#    for _ in range(n_permutations):
#        np.random.shuffle(pi)
#        
#        # Vectorized calculation for the current permutation
#        mij = symmetric_matrix * (pi[:, None] == pi) * a00
#        mij += symmetric_matrix * (pi[:, None] != pi) * a01
#        current_stat = np.sum(mij) / 2
#
#        if statistic <= current_stat:
#            larger += 1
#
#    return larger / n_permutations
#
#class MMDStatistic_Numpy:
#    r"""The *unbiased* MMD test of :cite:`gretton2012kernel`.
#
#    The kernel used is equal to:
#
#    .. math ::
#        k(x, x') = \sum_{j=1}^k e^{-\alpha_j\|x - x'\|^2},
#
#    for the :math:`\alpha_j` proved in :py:meth:`~.MMDStatistic.__call__`.
#
#    Arguments
#    ---------
#    n_1: int
#        The number of points in the first sample.
#    n_2: int
#        The number of points in the second sample."""
#
#    def __init__(self, n_1, n_2):
#        self.n_1 = n_1
#        self.n_2 = n_2
#
#        # The three constants used in the test.
#        self.a00 = 1. / (n_1 * (n_1 - 1))
#        self.a11 = 1. / (n_2 * (n_2 - 1))
#        self.a01 = - 1. / (n_1 * n_2)
#
#    def __call__(self, sample_1, sample_2, alphas, ret_matrix=False):
#        r"""Evaluate the statistic.
#
#        The kernel used is
#
#        .. math::
#
#            k(x, x') = \sum_{j=1}^k e^{-\alpha_j \|x - x'\|^2},
#
#        for the provided ``alphas``.
#
#        Arguments
#        ---------
#        sample_1: numpy array
#            The first sample, of size ``(n_1, d)``.
#        sample_2: numpy array
#            The second sample, of size ``(n_2, d)``.
#        alphas : list of float
#            The kernel parameters.
#        ret_matrix: bool
#            If set, the call with also return a second variable.
#
#            This variable can be then used to compute a p-value using
#            :py:meth:`~.MMDStatistic.pval`.
#
#        Returns
#        -------
#        float
#            The test statistic.
#        numpy array
#            Returned only if ``ret_matrix`` was set to true."""
#        sample_12 = np.vstack((sample_1, sample_2))
#        distances = squareform(pdist(sample_12, 'sqeuclidean'))
#
#        kernels = None
#        for alpha in alphas:
#            kernels_a = np.exp(- alpha * distances)
#            if kernels is None:
#                kernels = kernels_a
#            else:
#                kernels = kernels + kernels_a
#
#        k_1 = kernels[:self.n_1, :self.n_1]
#        k_2 = kernels[self.n_1:, self.n_1:]
#        k_12 = kernels[:self.n_1, self.n_1:]
#
#        mmd = (2 * self.a01 * k_12.sum() +
#               self.a00 * (k_1.sum() - np.trace(k_1)) +
#               self.a11 * (k_2.sum() - np.trace(k_2)))
#        if ret_matrix:
#            return mmd, kernels
#        else:
#            return mmd
#
#    def pval(self, distances, n_permutations=500): # the standard is 1000 but that might take forever
#        r"""Compute a p-value using a permutation test.
#
#        Arguments
#        ---------
#        distances: numpy array
#            The distances computed using :py:meth:`~.MMDStatistic.__call__`.
#        n_permutations: int
#            The number of random draws from the permutation null.
#
#        Returns
#        -------
#        float
#            The estimated p-value."""
#        return permutation_test_mat(distances, self.n_1, self.n_2, n_permutations, a00=self.a00, a11=self.a11, a01=self.a01)


class MMDCalculator(NumericBaseDriftCalculator):
    name = "mmd"

    def __init__(self, q_val=0.1, alternative='two-sided', mode='asymp', average='macro', include_critical_value=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.q_val = q_val
        self.alternative = alternative
        self.mode = mode
        self.average = average
        self.include_critical_value = include_critical_value


    
    def convert(self, arg):
        return arg

    def _predict(self, sample):

        from alibi_detect.cd import MMDDrift  # Move the import here

        np.random.seed(42)
        if len(self._ref) > 500:
            random_indices = np.random.choice(len(self._ref), 500, replace=False)
            limited_ref = self._ref.iloc[random_indices]
        else:
            limited_ref = self._ref

        nref = len(limited_ref)
        nobs = len(sample)
        out = {}


        # Flatten the nested lists in self._ref and sample
        flattened_ref = [item for sublist in limited_ref.tolist() for item in sublist]
        flattened_sample = [item for sublist in sample.tolist() for item in sublist]

        # Convert to numpy arrays
        ref_array = np.array(flattened_ref).reshape((nref, -1))
        sample_array = np.array(flattened_sample).reshape((nobs, -1))

        mmd_test_numpy = MMDDrift(ref_array, backend='pytorch')
        # As per the original MMD paper, the median distance between all points in the aggregate sample from both
        # distributions is a good heuristic for the kernel bandwidth, which is why compute this distance here.
        try:
            output = mmd_test_numpy.predict(sample_array, return_p_val=True, return_distance=True)
            p_val = output['data']['p_val']
            distance = output['data']['distance']
            distance_threshold = output['data']['distance_threshold']

            out["distance"], out['pval'] = distance, p_val

        except TypeError:
            out["distance"], out['pval'] = float("NaN"), float("NaN")

        if self.include_critical_value:
            raise NotImplementedError("Critical value not implemented for MMD")

        return out



class BasicDriftCalculator(NumericBaseDriftCalculator):
    name = "stats"

    def convert(self, arg):
        return pd.to_numeric(arg, errors="coerce")

    def _predict(self, sample):
        sample = pd.to_numeric(sample, errors="coerce")
        return {
            "mean": np.mean(sample),
            "std": np.std(sample),
            "median": np.median(sample)
        }
