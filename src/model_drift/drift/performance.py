#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import json

import numpy as np
import pandas as pd
import six
import torch
from sklearn import metrics
from torchmetrics.functional import auroc

from model_drift.drift.base import BaseDriftCalculator


def toarray(value):
    def _toarray(s):
        if isinstance(s, six.string_types):
            if "," not in s:
                s = ",".join(s.split())
            return np.array(json.loads(s))
        else:
            return np.array(s)

    return _toarray(value)


def macro_auc(scores, labels, skip_missing=True):
    if len(scores) == 0:
        return float('NaN')
    N = labels.shape[1]
    aucs = [0] * N
    for i in range(N):
        try:
            aucs[i] = auroc(torch.tensor(scores[i]), torch.tensor(labels[i]).long()).numpy()
        except Exception as e:
            if "No positive samples in targets" not in str(e):
                raise
            aucs[i] = float('NaN')

    aucs = np.array(aucs)
    c = (~np.isnan(aucs)).sum() if skip_missing else N
    return np.nansum(aucs) / c


def micro_auc(scores, labels):
    return float(auroc(torch.tensor(scores), torch.tensor(labels).long(), average='micro').numpy())

def youden_point(df, target_names):
    """
    Calculate the Youden Index for each disease and return that as optimal operating point for the disease.
    """
    operating_points = {}
    for disease in target_names:
        fpr, tpr, thresholds = metrics.roc_curve(df[f'label.{disease}'], df[f'activation.{disease}'])

        # Calculate Youden Index
        idx = np.argmax(tpr - fpr)
        best_cutoff = thresholds[idx]

        operating_points[disease] = format(best_cutoff, ".4f")

    return operating_points


def classification_report(scores, labels, target_names=None, th=None):
    keeps = (labels.sum(axis=0) > 0)

    if target_names is None:
        target_names = [str(i) for i in range(scores.shape[1])]
    target_names = np.array(target_names)

    if not isinstance(th, dict):
            raise ValueError("Thresholds (th) must be provided as a dictionary with target names as keys.")

    #binarize scores according to their thresholds
    binary_scores = np.zeros_like(scores, dtype=bool)
    for i, target in enumerate(target_names):
        if target in th:
            binary_scores[:, i] = scores[:, i] >= float(th[target])
        else:
            raise KeyError(f"No threshold provided for target '{target}'.")

    output = metrics.classification_report(labels, binary_scores, target_names=target_names, output_dict=True)
    for i, k in enumerate(target_names):
        if keeps[i] == 0:
            continue
        try:
            output[k]['auroc'] = metrics.roc_auc_score(
                labels[:, i], scores[:, i])
        except: # noqa
            return

    output['macro avg']['auroc'] = (metrics.roc_auc_score(
        labels[:, keeps], scores[:, keeps], labels=target_names[keeps], average='macro'))
    output['micro avg']['auroc'] = (metrics.roc_auc_score(labels, scores, average='micro'))

    return output


class AUROCCalculator(BaseDriftCalculator):
    name = "auroc"

    def __init__(self, label_col=None, score_col=None, average='micro', ignore_nan=True):
        self.label_col = label_col
        self.score_col = score_col
        self.average = average
        self.ignore_nan = ignore_nan

    def convert(self, arg):
        if not isinstance(arg, pd.DataFrame):
            raise NotImplementedError("only Dataframes supported")
        return arg.applymap(toarray)

    def _predict(self, sample):
        labels = sample.iloc[:, 1] if self.label_col is None else sample[self.label_col]
        scores = sample.iloc[:, 0] if self.score_col is None else sample[self.score_col]
        labels = np.stack(labels.values)
        scores = np.stack(scores.values)

        if self.average == "macro":
            return macro_auc(scores, labels)
        return micro_auc(scores, labels)


class ClassificationReportCalculator(BaseDriftCalculator):
    name = "class_report"

    def __init__(self, label_col=None, score_col=None, target_names=None, th=None):
        super().__init__()
        self.label_col = label_col
        self.score_col = score_col
        self.target_names = target_names
        self.th = th
    
    def prepare(self, ref):
        self._ref = self.convert(ref)
        if 'activation' in self._ref and 'label' in self._ref:

            activations_df = pd.DataFrame(self._ref['activation'].tolist(), index=self._ref.index)
            labels_df = pd.DataFrame(self._ref['label'].tolist(), index=self._ref.index)
            combined_df = pd.concat([activations_df, labels_df], axis=1)

            combined_df.columns = ['activation.' + str(k) for i, k in enumerate(self.target_names)] + \
                                ['label.' + str(k) for i, k in enumerate(self.target_names)]
            
            self._ref_df = combined_df
            self._is_prepared = True
        else:
            print("Error: 'ref' does not contain required 'activation' and 'label' columns.")

        self.th = youden_point(self._ref_df, self.target_names)
        self._is_prepared = True

    def convert(self, arg):
        if not isinstance(arg, pd.DataFrame):
            raise NotImplementedError("only Dataframes supported")
        return arg.applymap(toarray)

    def _predict(self, sample):
        labels = sample.iloc[:, 1] if self.label_col is None else sample[self.label_col]
        scores = sample.iloc[:, 0] if self.score_col is None else sample[self.score_col]

        labels = np.stack(labels.values)
        scores = np.stack(scores.values)

        return classification_report(scores, labels, target_names=self.target_names, th=self.th)
