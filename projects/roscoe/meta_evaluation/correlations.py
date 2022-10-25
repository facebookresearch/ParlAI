#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas
from scipy.stats import somersd


class Analyzer:
    def __init__(self, reference_column: str):
        self.reference_column = reference_column
        self.statistic = "Somers\' D"
        self.pval = "Somers\' D p-value"

    def correlation_stat(self, df, hypo_column):
        # N/A values should not be accounted for in correlations
        subset = df[[self.reference_column, hypo_column]].dropna()
        # It's very important that the human label is first for Somers' D!
        return somersd(subset[self.reference_column], subset[hypo_column])

    def get_statistics(self, df):

        somersd_statistic = []
        somersd_pval = []
        columns = []
        for column in df:
            if column != self.reference_column:
                columns.append(column)
                corr = self.correlation_stat(df=df, hypo_column=column)
                somersd_statistic.append(corr.statistic)
                somersd_pval.append(corr.pvalue)

        return {
            self.statistic: pandas.DataFrame(
                somersd_statistic, columns=[self.reference_column], index=columns
            ),
            self.pval: pandas.DataFrame(
                somersd_pval, columns=[self.reference_column], index=columns
            ),
        }
