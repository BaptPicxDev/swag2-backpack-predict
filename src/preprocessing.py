# Python modules.
from typing import (
    Dict,
    Text,
    Tuple,
    Union,
)


# Other modules.
import pandas as pd
from pandas.api.types import (
    is_numeric_dtype,
    is_object_dtype,
    is_bool_dtype,
    is_datetime64_any_dtype,
)
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    StandardScaler,
    PowerTransformer,  # Combat skewness
    PolynomialFeatures,
)
from sklearn.model_selection import train_test_split


# Functions.
def fill_df_navalues(df: pd.DataFrame) -> pd.DataFrame:
    """Fill na values depending on type.
    Super simple rules.

    :param df:
    :return pd.DataFrame:
    :note:
    -
    """
    # Creating a copy for safety.
    df_filled = df.copy()
    # Listing column_names
    for column_name in df.columns:
        if is_object_dtype(df_filled[column_name]):
            df_filled[column_name] = df_filled[column_name].fillna(value=df_filled[column_name].mode()[0])
        elif is_numeric_dtype(df_filled[column_name]):
            df_filled[column_name] = df_filled[column_name].fillna(value=df_filled[column_name].mean())
        elif is_bool_dtype(df_filled[column_name]):
            df_filled[column_name] = df_filled[column_name].fillna(value=df_filled[column_name].mode()[0])
        elif is_datetime64_any_dtype(df_filled[column_name]):
            min_date = df_filled[column_name].min()
            fill_date = min_date if pd.notna(min_date) else pd.Timestamp.today()
            df_filled[column_name] = df_filled[column_name].fillna(value=fill_date)
        else:
            raise TypeError(f"Wrong type: {df_filled[column_name].dtype}")
    return df_filled


def scale_and_encoder_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[Text, Union[LabelEncoder, MinMaxScaler]]]:
    """Scale the features.

    :param df:
    :return Tuple:
    :note:
    - Handle PowerTransformer
    """
    # Preparing output.
    encoders_and_scalers = {}
    # Creating a copy.
    scaled_and_encoded_df = df.copy()
    # Iterating over features.
    for column_name in scaled_and_encoded_df.columns:
        # Skipping identifier.
        if column_name == "id":
            continue
        # Preparing encoder & scaler.
        elif scaled_and_encoded_df[column_name].dtype in ["object", "category"]:
            enc = LabelEncoder()
            scaled_and_encoded_df[column_name] = enc.fit_transform(scaled_and_encoded_df.loc[:, [column_name]].values.ravel())
        elif scaled_and_encoded_df[column_name].dtype == "float64":
            enc = StandardScaler()
            scaled_and_encoded_df[column_name] = enc.fit_transform(scaled_and_encoded_df.loc[:, [column_name]].to_numpy())
        else:
            raise TypeError(f"{scaled_and_encoded_df[column_name].dtype}")
        # Fit the encoder and transforming the feature.
        encoders_and_scalers[column_name] = enc
    return scaled_and_encoded_df, encoders_and_scalers


def create_polynomial_features(df: pd.DataFrame, polynomial_degree=2) -> pd.DataFrame:
    """
    """
    new_df = df.copy()
    poly = PolynomialFeatures(polynomial_degree)
    new_df = poly.fit_transform(new_df.drop(columns=["index", "id"], errors="ignore"))
    return new_df


def split_X_y_in_train_test_sets(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into Train & Test sets.

    :param X:
    :param y:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test
