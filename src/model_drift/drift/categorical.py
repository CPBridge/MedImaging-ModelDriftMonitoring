#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from collections import Counter
import math

import numpy as np
from scipy.stats import chi2_contingency, chi2

from model_drift.drift.base import BaseDriftCalculator

def merge_freqs(ref_counts, sample):
    sample_counts = Counter(sample)
    keys = set().union(ref_counts, sample_counts)
    exp = np.array([ref_counts.get(k, 0) for k in keys])
    obs = np.array([sample_counts.get(k, 0) for k in keys])
    return exp, keys, obs

def hellinger_distance(p, q):
    """Hellinger distance between two discrete distributions. 
    """
    return math.sqrt(sum([ (math.sqrt(p_i) - math.sqrt(q_i))**2 for p_i, q_i in zip(p, q) ]) / 2)


class ChiSqDriftCalculator(BaseDriftCalculator):
    name = "chi2"

    def __init__(self, q_val=0.1, correction=True, lambda_=None, use_freq=False, include_critical_values=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.q_val = q_val
        self.correction = correction
        self.lambda_ = lambda_
        self.use_freq = use_freq
        self.include_critical_values = include_critical_values

    def convert(self, arg):
        return arg.apply(str)

    def prepare(self, ref, **kwargs):
        self._ref_counts = Counter(ref)
        super().prepare(ref)

    def _predict(self, sample):
        exp, keys, obs = merge_freqs(self._ref_counts, sample)

        if self.use_freq:
            exp = exp / exp.sum()
            obs = obs / obs.sum()

        out = {}
        out['distance'], out['pval'], dof, _ = chi2_contingency(np.vstack([exp, obs]),
                                                                correction=self.correction,
                                                                lambda_=self.lambda_)
        if self.include_critical_values:
            out['critical_value'] = chi2.ppf(1 - self.q_val, dof)
            out['critical_diff'] = out['distance'] - out['critical_value']

        return out



class ChiSqDriftCalculatorJackKnife(BaseDriftCalculator):
    name = "chi2_jackknife"

    def __init__(self, q_val=0.1, correction=True, lambda_=None, use_freq=False, include_critical_values=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.q_val = q_val
        self.correction = correction
        self.lambda_ = lambda_
        self.use_freq = use_freq
        self.include_critical_values = include_critical_values

    def convert(self, arg):
        return arg.apply(str)

    def prepare(self, ref, **kwargs):
        self._ref_counts = Counter(ref)
        super().prepare(ref)

    def _predict(self, sample):

        nref = len(self._ref)
        nobs = len(sample)

        ref1 = np.random.choice(self._ref, nobs)
        ref2 = np.random.choice(self._ref, nobs)

        ref1_counts = Counter(ref1)
        ref2_counts = Counter(ref2)

        exp_ref1_sam, keys, obs_ref1_sam = merge_freqs(ref1_counts, sample)

        exp_ref1_ref2, keys, obs_ref1_ref2 = merge_freqs(ref1_counts, ref2_counts)

        if self.use_freq:
            exp_ref1_sam = exp_ref1_sam / exp_ref1_sam.sum()
            obs_ref1_sam = obs_ref1_sam / obs_ref1_sam.sum()

            exp_ref1_ref2 = exp_ref1_ref2 / exp_ref1_ref2.sum()
            obs_ref1_ref2 = obs_ref1_ref2 / obs_ref1_ref2.sum()


        out = {}

        dist1, _, _, _ = chi2_contingency(np.vstack([exp_ref1_sam, obs_ref1_sam]),
                                                                correction=self.correction,
                                                                lambda_=self.lambda_)
        
        dist2, _, _, _  = chi2_contingency(np.vstack([exp_ref1_ref2, obs_ref1_ref2]),
                                                                correction=self.correction,
                                                                lambda_=self.lambda_)   
        
        out["distance"] = max(dist1 - dist2, 0.0)
        out['pval'] = float("NaN")

        
        if self.include_critical_values:
            raise NotImplementedError("Critical value not implemented for jackknife")

        return out
    
class HellingerDriftCalculatorJackKnife(BaseDriftCalculator):
    name = "hellinger_jackknife"

    def __init__(self, q_val=0.1, correction=True, lambda_=None, use_freq=True, include_critical_values=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.q_val = q_val
        self.correction = correction
        self.lambda_ = lambda_
        self.use_freq = use_freq
        self.include_critical_values = include_critical_values

    def convert(self, arg):
        return arg.apply(str)

    def prepare(self, ref, **kwargs):
        self._ref_counts = Counter(ref)
        super().prepare(ref)

    def _predict(self, sample):

        nref = len(self._ref)
        nobs = len(sample)

        ref1 = np.random.choice(self._ref, nobs)
        ref2 = np.random.choice(self._ref, nobs)

        ref1_counts = Counter(ref1)
        ref2_counts = Counter(ref2)

        exp_ref1_sam, keys, obs_ref1_sam = merge_freqs(ref1_counts, sample)

        exp_ref1_ref2, keys, obs_ref1_ref2 = merge_freqs(ref1_counts, ref2_counts)

        if self.use_freq:
            exp_ref1_sam = exp_ref1_sam / exp_ref1_sam.sum()
            obs_ref1_sam = obs_ref1_sam / obs_ref1_sam.sum()

            exp_ref1_ref2 = exp_ref1_ref2 / exp_ref1_ref2.sum()
            obs_ref1_ref2 = obs_ref1_ref2 / obs_ref1_ref2.sum()

        out = {}

        dist1 = hellinger_distance(exp_ref1_sam, obs_ref1_sam)
        dist2 = hellinger_distance(exp_ref1_ref2, obs_ref1_ref2)
 
        
        out["distance"] = max(dist1 - dist2, 0.0)
        out['pval'] = float("NaN")

        
        if self.include_critical_values:
            raise NotImplementedError("Critical value not implemented for jackknife")

        return out
#  