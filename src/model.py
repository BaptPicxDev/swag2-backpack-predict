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
)
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Functions.
def get_linear_regression_model() -> Tuple[Ridge, Dict]:
    """Regression model.

    :return Ridge: model linear regressio nRidge
    """
    grid_search_parameters = {
        "alpha": [0.01, 0.1, 1, 10], # Regularization strength.
    }
    return Ridge(), grid_search_parameters


def get_random_forest_model(random_state=42) -> Tuple[RandomForestRegressor, List]:
    """Regression model.

    :return RandomForestRegressor:
    """
    grid_search_parameters = {
        "n_estimators": [50, 100, 200],  # Number of trees.
        "max_depth": [5, 10],  # Tree depth.
    }
    return RandomForestRegressor(random_state=random_state), grid_search_parameters


def get_xgboost_model(random_state=42) -> Tuple[XGBRegressor, Dict]:
    """Regression model.

    :return XGBRegressor:
    """
    grid_search_parameters = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
    }
    return XGBRegressor(random_state=random_state), grid_search_parameters


def run_grid_search_and_kfold(
        model: Union[Ridge, RandomForestRegressor, XGBRegressor],
        parameters: Dict[Text, List],
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        number_of_split=2,
        scoring_method="r2",
        random_state=42,
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
    - 3hours for RandomForestRegressor/
    """
    # KFold(n_splits=number_of_split, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(model, parameters, cv=number_of_split, scoring=scoring_method)
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
 