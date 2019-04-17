# -*- coding: utf-8 -*-

import logging
import tempfile
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from pyramid.arima import auto_arima
import numpy as np

from mlprimitives.utils import import_object

LOGGER = logging.getLogger(__name__)


class ARIMAclass(object):
    """A Wrapper for the statsmodels ARIMA model for time series predictions"""

    def __init__(self, **hyperparameters):
        self.hyperparameters = hyperparameters

    def fit(self, X, index, **kwargs):
        X_conc = np.column_stack((index, X.flatten()))
        X_conc = pd.DataFrame(X_conc)
        X_conc = X_conc.sort_values(0).set_index(0)
        print("Fitting model")
        """
        for p in range(12):
            for d in range(12):
                for q in range(12):
                    try:
                        model = ARIMA(X_conc.values, order=(p, d, q))
                        model = model.fit(disp=0)
                        print('p: {}, d: {}, q: {}, aic: {}'.format(p, d, q, model.aic))
                    except:
                        pass
        """
        self.model = ARIMA(X_conc.values, order=(8, 0, 9))
        #self.model = auto_arima(X)
        self.model = self.model.fit(disp=0)
        print(self.model.aic)

    def predict(self, X):
        print("Predicting values")
        print(len(X))
        #y = self.model.predict(len(X))
        y = self.model.forecast(len(X))
        print(y[0])
        return y[0]
