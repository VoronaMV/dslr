import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod


class LogisticRegression(metaclass=ABCMeta):

    def __init__(self, df: pd.DataFrame = None):
        self.frame = df

    @property
    def frame(self):
        if self.__frame is None:
            return NotImplemented
        return self.__frame

    @frame.setter
    def frame(self, df: pd.DataFrame):
        if df is not None and not isinstance(df, pd.DataFrame):
            raise Exception('Wrong format. Should be pd.DataFrame object')
        self.__frame = None
        if df is not None:
            self.__frame = df.copy()

    @property
    def y(self):
        if self.__y is None:
            return NotImplemented
        return self.__y

    @y.setter
    def y(self, y):
        self.__y = y

    @property
    def X(self):
        if self.__X is None:
            return NotImplemented
        return self.__X

    @X.setter
    def X(self, X):
        if X is not None and not isinstance(X, pd.DataFrame):
            raise Exception('Wrong format. Should be pd.DataFrame object')
        self.__X = X

    @property
    def theta(self):
        if self.__theta is None:
            return NotImplemented
        return self.__theta

    @theta.setter
    def theta(self, theta):
        if theta is not None and not isinstance(theta, pd.DataFrame):
            raise Exception('Wrong format. Should be pd.DataFrame object')
        self.__theta = theta

    def set_target_column(self, column_name: str):
        """
        According target column and set X, y, theta values.
        """
        columns = list(self.frame.columns.values)
        if column_name not in columns:
            raise Exception(f'No <{column_name}> column in dataframe')
        indx = columns.index(column_name)
        columns.pop(indx)
        self.y = pd.DataFrame(self.frame[column_name])
        self.X = pd.DataFrame(self.frame[columns])

        theta_names = [f'theta_{col}' for col in columns]
        theta_shape = (1, len(theta_names))
        self.theta = pd.DataFrame(data=np.zeros(theta_shape), columns=theta_names)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def cost_gradient(self, theta, X, y):
        predictions = self.sigmoid(X @ theta)
        return X.T @ (predictions - y) / len(y)

    def cost(self, theta, X, y):
        predictions = self.sigmoid(X @ theta)
        predictions[predictions >= 0.5] = 0.99999999
        predictions[predictions < 0.5] = 0.000000001
        # print(predictions[predictions != 1])
        # predictions[predictions == 1] = 0.999
        error = -y * np.log(predictions) - (1 - y) * np.log(1 - predictions)
        return sum(error) / len(y)

    @staticmethod
    @abstractmethod
    def fit(cost: callable, initial_theta, cost_gradient: callable, X, y):
        """
        Minimize a function using a gradient algorithm.
        return: Vector of result weights for the model
        """
        # TODO: Gradient algorithm should be implemented
        pass

    @classmethod
    @abstractmethod
    def predict(cls, theta, X, y):
        pass


class LogisticRegressionBinary(LogisticRegression):

    @staticmethod
    def fit(cost: callable, initial_theta, cost_gradient, X, y):
        """
        Minimize a function using a gradient algorithm.
        return: Vector of result weights for the model
        """
        stable_cost_diff = 0

        prev_cost = cost(initial_theta, X, y)
        prev_theta = initial_theta
        while True:
            theta = prev_theta - 0.1 * cost_gradient(prev_theta, X, y)
            current_cost = cost(theta, X, y)
            print('cost_diff=', abs(current_cost - prev_cost))

            cost_diff = abs(current_cost - prev_cost)
            if current_cost > prev_cost:
                print('current_cost > prev_cost')
                break
            elif stable_cost_diff == 1000:
                break
            elif cost_diff <= 0.00000000001:
                stable_cost_diff += 1
            prev_cost = current_cost
            prev_theta = theta

        return theta

    @classmethod
    def predict(cls, theta, X, y):
        predictions = cls.sigmoid(X @ theta)
        predictions[predictions >= 0.5] = 1.0
        predictions[predictions < 0.5] = 0.0
        return y == predictions


class LogisticRegressionMultinomial(LogisticRegression):

    def set_target_column(self, column_names: list):
        # TODO: It similar for parent class method. May be combine them.
        """
        According target column and set X, y, theta values.
        """
        columns = list(self.frame.columns.values)
        for column_name in column_names:
            if column_name not in columns:
                raise Exception(f'No <{column_name}> column in dataframe')
            indx = columns.index(column_name)
            columns.pop(indx)
        self.y = pd.DataFrame(self.frame[column_names])
        self.X = pd.DataFrame(self.frame[columns])

        theta_names = [f'theta_{col}' for col in columns]
        theta_shape = (len(column_names), len(theta_names))
        self.theta = pd.DataFrame(data=np.zeros(theta_shape), columns=theta_names)

    def cost(self, theta, X, y):
        predictions = self.sigmoid(X @ theta)
        predictions[(predictions == predictions.max(axis=1)[:, None])] = 0.99999999
        predictions[(predictions != predictions.max(axis=1)[:, None])] = 0.000000001
        # print(predictions[predictions != 1])
        # predictions[predictions == 1] = 0.999
        error = -y * np.log(predictions) - (1 - y) * np.log(1 - predictions)
        return sum(error) / len(y)

    @staticmethod
    def fit(cost: callable, initial_theta, cost_gradient, X, y):
        """
        Minimize a function using a gradient algorithm.
        return: Vector of result weights for the model
        """
        stable_cost_diff = 0

        prev_cost = cost(initial_theta, X, y)
        prev_theta = initial_theta
        while True:
            theta = prev_theta - 0.1 * cost_gradient(prev_theta, X, y)
            current_cost = cost(theta, X, y)
            print('cost_diff=', abs(current_cost - prev_cost))

            cost_diff = abs(current_cost - prev_cost)
            if np.sum(current_cost) > np.sum(prev_cost):
                print('current_cost > prev_cost')
                break
            elif stable_cost_diff == 1000:
                break
            elif np.sum(cost_diff) <= 0.00000000001:
                stable_cost_diff += 1
            prev_cost = current_cost
            prev_theta = theta

        return theta

    @classmethod
    def predict(cls, theta, X, y):
        predictions = cls.sigmoid(X @ theta)
        predictions[(predictions == predictions.max(axis=1)[:, None])] = 1.0
        predictions[(predictions != predictions.max(axis=1)[:, None])] = 0.0
        return y == predictions
