from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


def score(clf, x, y):
    print(y.shape, x.shape)
    a = clf.predict_proba(x)
    print('predict_proba shape' , a.shape)
    return roc_auc_score(y == 1, clf.predict_proba(x))


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params
        
        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        si = -self.loss_derivative(y=y, z=predictions)
        model = self.base_model_class(**self.base_model_params).fit(x, si)
        gamma = self.find_optimal_gamma(y, predictions, model.predict(x))
        self.gammas.append(gamma)
        self.models.append(model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        for _ in range(self.n_estimators):
            print('next loop iteration')
            self.fit_new_base_model(x_train, y_train, train_predictions)
            print('fitted_new_base')
            
            train_predictions = self.predict_proba(x_train)
            print('predicted_train')
            valid_predictions = self.predict_proba(x_valid)
            print('predicted_valid')

            self.history['history'].append(self.score(x_valid, y_valid))

            if self.early_stopping_rounds is not None:
                if self.history['history'][-1] >= self.history['history'][-2]:
                    self.early_stopping_rounds -= 1
                if self.early_stopping_rounds == 0:
                    break

        if self.plot:
            # TODO
            pass

    def predict_proba(self, x):
        a_x = np.zeros((x.shape[0], 1))
        print(self.gammas)
        print(self.models)
        print(x.shape, type(x))
        for gamma, model in zip(self.gammas, self.models):
            a_x += np.array([model.predict(t) for t in x]) * gamma
        ans = np.array([self.sigmoid(t) for t in a_x])
        return ans

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        pass
