#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.special import kolmogi
from scipy.stats import ks_2samp
import ot

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


class EMDDriftCalculatorJackKnife(NumericBaseDriftCalculator):
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
