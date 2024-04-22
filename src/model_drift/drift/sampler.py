#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import numpy as np
from sklearn.utils import resample


class Sampler(object):
    def __init__(self, sample_size, replacement=True, random_state=None):
        self.sample_size = sample_size
        self.replacement = replacement
        self.random_state = random_state

    def sample_index(self, index, stratify=None):
        if not self.replacement and len(index) < self.sample_size:
            return np.array(index)
        return resample(index, n_samples=self.sample_size, replace=self.replacement, random_state=self.random_state,
                        stratify=stratify)

    def sample(self, sample, stratify=None):
        return sample[self.sample_index(range(len(sample)), stratify=stratify)]

    def sample_iterator(self, sample, n_samples=1, stratify=None):
        for _ in range(n_samples):
            yield self.sample(sample, stratify=stratify)

class DummySampler(object):
    """
    This is a dummy sampler that mimics the behavior of the Sampler class, but does
    not perform any samples and instead returns the indicies unchanged. This is used
    for the Flapjack resampling, in which the reference dataframe is resampled in the 
    metric itself and the sliding window sample is not resampled at all. In this case 
    we still want to create multiple copies of the sample window. 
    """
    def __init__(self, sample_size=0, replacement=True, random_state=None):
        # These variables are all dummy variables, just kept for future
        # compatibility. 
        self.sample_size = sample_size
        self.replacement = replacement
        self.random_state = random_state

    def sample_iterator(self, indices, n_samples=1, stratify=None):
      
        for _ in range(n_samples):
            yield indices
