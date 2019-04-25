# -*- coding: utf-8 -*-

import logging
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import numpy as np


LOGGER = logging.getLogger(__name__)


class ARIMAMODEL(object):
    """A Wrapper for the statsmodels ARIMA model for time series predictions"""

    def __init__(self, **hyperparameters):
        self.hyperparameters = hyperparameters

    def fit(self, X, index):
        X_conc = np.column_stack((index, X.flatten()))
        X_conc = pd.DataFrame(X_conc)
        X_conc = X_conc.sort_values(0).set_index(0)
        self.model = ARIMA(X_conc.values, order=(8, 0, 9))
        self.model = self.model.fit(disp=0)

    def predict(self, X):
        y = self.model.forecast(len(X))
        return y[0]
