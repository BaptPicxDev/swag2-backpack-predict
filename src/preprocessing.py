# Python modules.
from typing import (
    Dict,
    Text,
    Tuple,
    Union,
)


# Other modules.
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_numeric_dtype,
    is_object_dtype,
    is_bool_dtype,
    is_datetime64_any_dtype,
)
from sklearn.preprocessing import (
    OneHotEncoder,
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
            # enc = OneHotEncoder(sparse_output=False)
            # scaled_and_encoded_df[column_name] = enc.fit_transform(scaled_and_encoded_df.loc[:, [column_name]])
        elif scaled_and_encoded_df[column_name].dtype == "float64":
            enc = StandardScaler()
            scaled_and_encoded_df[column_name] = enc.fit_transform(scaled_and_encoded_df.loc[:, [column_name]].to_numpy())
        else:
            raise TypeError(f"{scaled_and_encoded_df[column_name].dtype}")
        # Fit the encoder and transforming the feature.
        encoders_and_scalers[column_name] = enc
    return scaled_and_encoded_df, encoders_and_scalers


def encode_categories_using_encoders_and_scalers(df: pd.DataFrame, encoders_and_scalers: Dict[Text, Union[LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler]]) -> pd.DataFrame:
    """Use encoder to properly encode the DataFrame columns using trained encoders and scalers.

    :param df:
    :param encoders_and_scalers:
    :return pd.DataFrame:
    """
    # Creating a copy.
    to_encode_df = df.copy()
    # Iterating over columns.
    for column_name in to_encode_df.columns:
        if column_name in ["id", "index"]:
            continue
        elif to_encode_df[column_name].dtype in ["object", "category"]:
            to_encode_df[column_name] = (
                encoders_and_scalers[column_name]
                .transform(to_encode_df.loc[:, [column_name]].values.ravel())
            )
        elif to_encode_df[column_name].dtype == "float64":
            to_encode_df[column_name] = (
                encoders_and_scalers[column_name]
                .transform(to_encode_df.loc[:, [column_name]].to_numpy())
            )
        else:
            raise TypeError(f"{to_encode_df[column_name].dtype}")
    return to_encode_df



def create_polynomial_features(df: pd.DataFrame, polynomial_degree=3) -> Tuple[pd.DataFrame, PolynomialFeatures]:
    """Use the PolynomialFeatures transformer to process input.

    :param df:
    :param polynomial_degree:
    :return Tuple:
    """
    new_df = df.copy().set_index(keys="id")
    indexes = np.asarray(new_df.index).astype(int)
    poly_enc = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
    polynomial_feature_values = poly_enc.fit_transform(new_df)
    new_df = pd.DataFrame(
        data=polynomial_feature_values,
        index=indexes,
        columns=["Original"] + [f"polynomial_{index}" for index in range(1, polynomial_feature_values.shape[1])],
    )
    return new_df.reset_index().rename(columns={"index": "id"}), poly_enc


def generate_polynomial_column_using_polynomial_feature_encoder(df: pd.DataFrame, polynomial_encoder: PolynomialFeatures) -> pd.DataFrame:
    """Use trained PolynomialFeatures to generate polynomial columns.

    :param df:
    :param polynomial_encoder:
    :return pd.DataFrame:
    """
    # Creating a copy.
    df_polynomial = df.copy().set_index(keys="id")
    indexes = np.asarray(df_polynomial.index).astype(int)
    polynomial_feature_values = polynomial_encoder.transform(df_polynomial)
    return pd.DataFrame(
        data=polynomial_feature_values,
        index=indexes,
        columns=["Original"] + [f"polynomial_{index}" for index in range(1, polynomial_feature_values.shape[1])],
    ).reset_index().rename(columns={"index": "id"})


def split_X_y_in_train_test_sets(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into Train & Test sets.

    :param X:
    :param y:
    :param test_size:
    :param random_state:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
