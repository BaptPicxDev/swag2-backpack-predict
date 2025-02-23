# Python modules.
from typing import (
    Dict,
    List,
    Text,
    Tuple,
    Union,
)


# Other modules.
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
)
from sklearn.metrics import (
    mean_squared_error,
    make_scorer,
)
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


# Functions.
def get_ridge_model() -> Tuple[Ridge, Dict]:
    """Regression model.

    :return Ridge: model linear regressio nRidge
    """
    grid_search_parameters = {
        "alpha": [
            0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
            0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            20, 50, 100, 500, 1000,
        ], # Regularization strength.
    }
    return Ridge(), grid_search_parameters


def get_lasso_model() -> Tuple[Lasso, Dict]:
    """Regression model.

    :return Lasso: model linear regressio nRidge
    """
    grid_search_parameters = {
        "alpha": [
            0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
            0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            20, 50, 100, 500, 1000,
        ], # Regularization strength.
    }
    return Lasso(), grid_search_parameters


def get_linear_regression_model() -> Tuple[LinearRegression, Dict]:
    """Regression model.

    :return LinearRegression: model linear regressio nRidge
    """
    grid_search_parameters = {
        "fit_intercept": [True, False],
    }
    return LinearRegression(), grid_search_parameters


def get_random_forest_model(random_state=42) -> Tuple[RandomForestRegressor, List]:
    """Regression model.

    :return RandomForestRegressor:
    """
    grid_search_parameters = {
        # Main parameters.
        "n_estimators": [5, 15, 50, 100, 150],  # Number of trees.
        "max_depth": [2, 4, 6],  # Tree depth.
        # Deep parameters.
        "max_features": ["auto", "sqrt", "log2"],
        "criterion" :["gini", "entropy"],
    }
    return RandomForestRegressor(random_state=random_state), grid_search_parameters


def get_xgboost_model(random_state=42) -> Tuple[XGBRegressor, Dict]:
    """Regression model.

    :return XGBRegressor:
    """
    grid_search_parameters = {
        # Main parameters.
        'n_estimators': [50, 100, 150, 200, 250],  # Number of trees.
        'max_depth': [3, 5, 7, 10],  # Tree depth.
        'learning_rate': [0.01, 0.1, 0.5],  # Step size at each iteration.
        # Deep parameters.
        'gamma': [0.5, 1.0],
        # 'min_child_weight': [1, 5], #, 10
        # 'subsample': [0.6, 1.0],
        # 'colsample_bytree': [0.6, 1.0],
    }
    return XGBRegressor(random_state=random_state), grid_search_parameters


def get_catboost_model(random_state=42) -> Tuple[CatBoostRegressor, Dict]:
    """Regression model.

    :return CatBoostRegressor:
    """
    grid_search_parameters = {
        # Main parameters.
        'depth': [4, 7, 10],  # Depth of trees.
        'iterations': [70, 200, 500],  # Number of boosting rounds.
        'l2_leaf_reg': [1, 5, 10],  # Regularization parameter.
        'learning_rate': [0.01, 0.05, 0.1],  # Step size at each iteration.
        # 'border_count': [32, 64, 128],
    }
    return CatBoostRegressor(random_state=random_state, verbose=0), grid_search_parameters


def get_lgbm_model(random_state=42) -> Tuple[LGBMRegressor, Dict]:
    """Regression model.

    :return LGBMRegressor:
    """
    grid_search_parameters = {
        # Main parameters.
        'n_estimators': [50, 100, 150, 200, 250],  # Number of trees.
        'max_depth': [3, 5, 7, 10, 12],  # Tree depth.
        'learning_rate': [0.1, 0.3, 0.5, 0.7],  # Step size at each iteration.
    }
    return LGBMRegressor(random_state=random_state, verbose=-1), grid_search_parameters


def run_grid_search(
        model: Union[Ridge, RandomForestRegressor, XGBRegressor, CatBoostRegressor, LGBMRegressor],
        parameters: Dict[Text, List],
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        number_of_split=3,
    ) -> Tuple[Union[Ridge, RandomForestRegressor, XGBRegressor], Dict[Text, Text]]:
    """Run KFold & GridSearch to find the best model.

    :param model:
    :param parameters:
    :param X_train:
    :param y_train:
    :param number_of_split:
    :return Union[Ridge, RandomForestRegressor, XGBRegressor]: The best model.
    """
    def rmse_scorer(y_true: pd.Series, y_pred: pd.Series) -> np.ndarray:
        """Custom scoring function that applies np.sqrt to MSE.
        To update function, depending on competition!

        :param y_true:
        :param y_pred:
        :return np.ndarray:
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))
    # Wrap it correctly for GridSearchCV
    custom_scorer = make_scorer(rmse_scorer, greater_is_better=False)
    # Setting up KFold, better than usual CV.
    kf = KFold(n_splits=number_of_split, shuffle=False)
    # Preparing GridSearch.
    grid_search = GridSearchCV(model, parameters, cv=kf, scoring=custom_scorer, verbose=3)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.cv_results_


if __name__ == "__main__":
    print("Just a simple test for Ridge")
    # Import
    from sklearn.datasets import fetch_california_housing
    # Data
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names).iloc[:1000]
    y = data.target[:1000]
    # Model
    model, parameters = get_random_forest_model()
    best = run_grid_search(model=model, parameters=parameters, X_train=X, y_train=y, number_of_split=2)
    print(type(best))
