# This file is part of the SPLINTER library.
# Copyright (C) 2012 Bjarne Grimstad (bjarne.grimstad@gmail.com).
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from . import splinter
from .function import Function
from .utilities import *
import numpy as np


class BSplineBoosting(Function):
    def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, alpha=0.9):
        """
        Class for stochastic gradient boosting with B-spline learners
        TODO: add init and warm_start as in sklearn.ensemble.GradientBoostingRegressor
        """
        super(Function, self).__init__()
        self._loss = loss  # Function pointer here
        self._learning_rate = learning_rate
        self._n_estimators = n_estimators
        self._subsample = subsample
        self._alpha = alpha
        self._learner = 'pspline'
        self._estimators = np.array((self._n_estimators,))
        self._oob_improvement = np.array((self._n_estimators,))
        self._train_score = np.array((self._n_estimators,))

    def fit(self, x, y):
        """
        Fit to data (x, y)
        """

    def predict(self, x):
        """
        Predict value at x
        """
        return 0


def sum_of_squares(a, b):
