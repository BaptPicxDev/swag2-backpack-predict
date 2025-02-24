# Python modules.
from collections import Counter
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
from pandas.api.types import (
    is_numeric_dtype,
    is_object_dtype,
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
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
def create_new_columns_based_on_categorical_columns(
    df: pd.DataFrame,
    categorical_columns: List[Text],
    output_col_name: Text,
) -> Tuple[pd.DataFrame, List[Text]]:
    """
    inpiration: https://www.kaggle.com/code/cdeotte/feature-engineering-with-rapids-lb-38-847

    :param df:
    :param output_col_name:
    :return Tuple[pd.DataFrame, List[Text]]:
    """
    # Create a copy for safety.
    new_df = df.copy()
    # Iteration over columns.
    for column_name in categorical_columns:
        if column_name in ["id", "index"]:
            continue
        elif column_name == output_col_name:
            continue
        new_cn = f"{column_name.lower().replace(" ", "")}_wc"
        new_df[new_cn] = (
            (
                new_df[column_name].astype("float32") * 100
                + new_df[output_col_name]
            ).astype("float32")
        )
    return new_df


def compute_statistical_df_focusing_on_output_and_specific_column(
    df: pd.DataFrame,
    column_to_focus_on: str,
    output_column_name: str,
    column_names_to_work_with: list,
    stats=["mean", "std", "count", "nunique", "median", "min", "max", "skew"],
    simple_stats=["mean", "std"],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create df stats mapping. Used to improve the inputs.
    Use the train dataset to generate the mapping.
    Then merge it with the test one to create meangfull columns.

    :param df:
    :param column_to_focus_on:
    :param output_colum_name:
    :param column_names_to_work_with:
    :param stats:
    :param simple_stats:
    :return Tuple[pd.DataFrame, pd.DataFrame]:
    :notes:
    - to verify
    """
    # Create a copy for safety.
    new_df = df.copy()
    # Computing partial variable name.
    partial_var_name = column_to_focus_on.lower().replace(" ", "_")
    # Generating the statistic related groupBy column. Feature engineering step 1.
    df_stats_mapping = new_df.groupby(column_to_focus_on)[output_column_name].agg(stats).astype('float32')
    # Renaming columns
    df_stats_mapping.columns = [f"TE1_{partial_var_name}_{output_column_name}_{stat}" for stat in stats]
    # Filling na.
    df_stats_mapping = fill_df_navalues(df=df_stats_mapping)
    # Merging
    new_df = pd.merge(
        new_df,
        df_stats_mapping.reset_index(),
        how="left",
        on=column_to_focus_on,
    )
    # Iterating over combo columns.
    for combo_column_name in column_names_to_work_with:
        # Feature engineering step 2.
        tmp_df = (
            new_df
            .groupby(combo_column_name)[output_column_name]
            .agg(simple_stats)
            .astype('float32')
        )
        tmp_df.columns = [f"TE2_{combo_column_name}_{output_column_name}_{cn}" for cn in simple_stats]
        new_df = pd.merge(
            new_df,
            tmp_df.reset_index(),
            on=combo_column_name,
            how="left",
        )
    # # Filling na.
    # new_df = fill_df_navalues(df=new_df)
    return new_df, df_stats_mapping.reset_index()


def remove_outliers(df: pd.DataFrame, threshold_number_of_outliers: int, outlier_step=1.5) -> List:
    """Takes a dataframe and returns an index list corresponding to the observations
    containing more than n outliers according to the Tukey IQR method.
    Inspiration: https://www.kaggle.com/code/marcinrutecki/gridsearchcv-kfold-cv-the-right-way

    :param df:
    :param threshold_number_of_outliers:
    :param outlier_step:
    :return pd.DataFrame:
    :notes:
    - to verify
    :example:
    >>> outliers = remove_outliers(df=df, threshold_number_of_outliers=1)
    >>> df = df.drop(outliers, axis=0).reset_index(drop=True)
    """
    # Create a copy for safety.
    new_df = df.copy()
    # Retrieving columns depending on type.
    column_names = [
        column_name for column_name in df.columns
        if is_numeric_dtype(df[column_name])
    ]
    # Preparing outputs.
    outlier_list = []
    for column_name in column_names:
        # 1st quartile (25%)
        Q1 = np.percentile(new_df[column_name], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(new_df[column_name], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        # outlier step
        outlier_step = outlier_step * IQR
        # Determining a list of indices of outliers
        outlier_list_column = (
            new_df[
                (
                    (new_df[column_name] < Q1 - outlier_step)
                    | (new_df[column_name] > Q3 + outlier_step )
                )
            ].index
        )
        # Extending outlier list.
        outlier_list.extend(outlier_list_column)
    # selecting observations containing more than x outliers
    outlier_list = Counter(outlier_list)
    multiple_outliers = list(
        key for key, value in outlier_list.items()
        if value > threshold_number_of_outliers
    )
    # Calculate the number of records below and above lower and above bound value respectively
    out1 = df[df[column_name] < Q1 - outlier_step]
    out2 = df[df[column_name] > Q3 + outlier_step]
    print(f"Total number of deleted outliers is: {out1.shape[0] + out2.shape[0]}.")
    # Returning df Without outliers.
    return multiple_outliers


def apply_log_to_output(df: pd.DataFrame, output_column_name: Text) -> pd.DataFrame:
    """Apply numpy log function to output column.

    :param df:
    :param output_column_name:
    :return pd.DataFrame:
    """
    # Verify type.
    if not is_numeric_dtype(df[output_column_name]):
        raise TypeError("")
    # Create a copy for safety.
    new_df = df.copy()
    # Generating output.
    new_df["log_" + output_column_name] = np.log1p(new_df[output_column_name].values.reshape(-1))
    return new_df


def reverse_log(df: pd.DataFrame, log_column_name: Text) -> pd.DataFrame:
    """Apply numpy exp to reverse log1p function applied previously.

    :param df:
    :param log_column_name:
    :return pd.DataFrame:
    """
    # Verify type.
    if not is_numeric_dtype(df[log_column_name]):
        raise TypeError("")
    # Create a copy for safety.
    new_df = df.copy()
    # Generating output.
    new_df["reverse_" + log_column_name] = np.expm1(new_df[log_column_name].values.reshape(-1))
    return new_df


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
        if is_object_dtype(df_filled[column_name]) or (df_filled[column_name].dtype == "category"):
            df_filled[column_name] = df_filled[column_name].fillna(value=df_filled[column_name].mode()[0])
        elif is_numeric_dtype(df_filled[column_name]):
            df_filled[column_name] = df_filled[column_name].fillna(value=df_filled[column_name].median())
        elif is_bool_dtype(df_filled[column_name]):
            df_filled[column_name] = df_filled[column_name].fillna(value=df_filled[column_name].mode()[0])
        elif is_datetime64_any_dtype(df_filled[column_name]):
            min_date = df_filled[column_name].min()
            fill_date = min_date if pd.notna(min_date) else pd.Timestamp.today()
            df_filled[column_name] = df_filled[column_name].fillna(value=fill_date)
        else:
            raise TypeError(f"Wrong type: {df_filled[column_name].dtype}")
    return df_filled


def scale_and_encoder_features(
        df: pd.DataFrame,
        skip_column_names: List[Text],
    ) -> Tuple[pd.DataFrame, Dict[Text, Union[LabelEncoder, MinMaxScaler]]]:
    """Scale the features.

    :param df:
    :param skip_column_names:
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
        if (column_name == "id") or (column_name in skip_column_names):
            continue
        # Preparing encoder & scaler.
        elif scaled_and_encoded_df[column_name].dtype in ["object", "category"]:
            enc = LabelEncoder()
            scaled_and_encoded_df[column_name] = enc.fit_transform(scaled_and_encoded_df.loc[:, [column_name]].values.ravel())
            scaled_and_encoded_df[column_name] = scaled_and_encoded_df[column_name].astype("category")
            # enc = OneHotEncoder(sparse_output=False)
            # scaled_and_encoded_df[column_name] = enc.fit_transform(scaled_and_encoded_df.loc[:, [column_name]])
        elif scaled_and_encoded_df[column_name].dtype in ["float32", "float64"]:
            enc = StandardScaler()
            scaled_and_encoded_df[column_name] = enc.fit_transform(scaled_and_encoded_df.loc[:, [column_name]].to_numpy())
        else:
            raise TypeError(f"{scaled_and_encoded_df[column_name].dtype}")
        # Fit the encoder and transforming the feature.
        encoders_and_scalers[column_name] = enc
    return scaled_and_encoded_df, encoders_and_scalers


def encode_categories_using_encoders_and_scalers(
        df: pd.DataFrame,
        encoders_and_scalers: Dict[Text, Union[LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler]],
        skip_column_names=[],
    ) -> pd.DataFrame:
    """Use encoder to properly encode the DataFrame columns using trained encoders and scalers.

    :param df:
    :param encoders_and_scalers:
    :param skip_column_names:
    :return pd.DataFrame:
    """
    # Creating a copy.
    to_encode_df = df.copy()
    # Iterating over columns.
    for column_name in to_encode_df.columns:
        if column_name in ["id", "index"]:
            continue
        elif column_name in skip_column_names:
            continue
        elif to_encode_df[column_name].dtype in ["object", "category"]:
            to_encode_df[column_name] = (
                encoders_and_scalers[column_name]
                .transform(to_encode_df.loc[:, [column_name]].values.ravel())
            )
            to_encode_df[column_name] = to_encode_df[column_name].astype("category")
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


def get_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dummyfied DataFrame.
    Basically, each column present a category with yes or no (0 and 1).
    It allows models to easily differenciate -> better result.

    :param df:
    :return pd.DataFrame:
    :note:
    - ensure there is not too many distinct int, can fail otherwise, or take too long... 
    """
    # Listing int64 column types.
    categorical_column_names = [column_name for column_name in df.columns if df[column_name].dtype.name == "category"]
    return pd.get_dummies(
        data=df,
        prefix=categorical_column_names,
        columns=categorical_column_names,
        dtype=np.float32,
    )


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
