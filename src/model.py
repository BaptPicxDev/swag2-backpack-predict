# Python modules.
from typing import (
    Dict,
    List,
    Text,
    Tuple,
    Union,
)


# Other modules.
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
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

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
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.5],
        # Deep parameters.
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1.5, 5],
        'subsample': [0.6, 1.0],
        'colsample_bytree': [0.6, 1.0],
    }
    return XGBRegressor(random_state=random_state), grid_search_parameters


def run_grid_search_and_kfold(
        model: Union[Ridge, RandomForestRegressor, XGBRegressor],
        parameters: Dict[Text, List],
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        number_of_split=3,
        scoring_method="r2",
    ) -> Union[Ridge, RandomForestRegressor, XGBRegressor]:
    """Run KFold & GridSearch to find the best model.

    :param model:
    :param parameters:
    :param X_train:
    :param y_train:
    :param number_of_split:
    :param scoring_method:
    :param random_state:
    :return Union[Ridge, RandomForestRegressor, XGBRegressor]: The best model.
    :runtime:
    - 3 hours for RandomForestRegressor. at least....
    """
    # KFold(n_splits=number_of_split, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(model, parameters, cv=number_of_split, scoring=scoring_method, verbose=3)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


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
    best = run_grid_search_and_kfold(model=model, parameters=parameters, X_train=X, y_train=y, number_of_split=2)
    print(type(best))
 