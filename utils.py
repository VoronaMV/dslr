import pandas as pd
import numpy as np


class FrameHandler:

    @classmethod
    def cut_features(cls, df: pd.DataFrame, features=[]) -> pd.DataFrame:
        return df.drop(columns=features, inplace=False)

    @classmethod
    def normalize_data(cls, df: pd.DataFrame, columns=[]) -> pd.DataFrame:
        """
        Normalize all columns data if colums argument doesn't set.
        Otherwice - normizes only defined columns.
        """
        normalized_df = df.copy()
        if columns:
            normalized_df[columns] = cls.__normalize(normalized_df[columns])
        else:
            normalized_df = cls.__normalize(normalized_df)
        return normalized_df

    @classmethod
    def __normalize(cls, df):
        return (df - df.min()) / (df.max() - df.min())

    @classmethod
    def filter_numeric(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df._get_numeric_data()

    @classmethod
    def prepend_ones(cls, df: pd.DataFrame, column_name: str = 'bias') -> pd.DataFrame:
        rows, _ = df.shape
        ones = np.ones(rows)
        bias_df = pd.DataFrame({column_name: ones}, index=df.index)
        concatenated_df = pd.concat([bias_df, df], axis=1)
        return concatenated_df


def prepare_dataframe(df: pd.DataFrame, drop_features=[], normalize=True) -> pd.DataFrame:
    """
    Prepare dataframe for model:
        - drop features were received
        - create dummies
        - filter only numeric values
        - clean datafrate from NaN values
        - normalize data if "normalize" is True
    """
    prepared_df = FrameHandler.cut_features(df, drop_features)

    # create dummie variables for hands and houses
    # TODO: Investigate. I thing its much better without it
    # hand_dummies_df = pd.get_dummies(prepared_df['Best Hand'])
    house_dummies_df = pd.get_dummies(prepared_df['Hogwarts House'])
    # prepared_df = pd.concat([prepared_df, hand_dummies_df, house_dummies_df], axis=1)
    prepared_df = pd.concat([prepared_df, house_dummies_df], axis=1)

    # filter only numeric columns and drop NaN
    prepared_df = FrameHandler.filter_numeric(prepared_df)
    prepared_df.dropna(how='any', inplace=True)

    # normalize data
    if normalize:
        prepared_df = FrameHandler.normalize_data(prepared_df)

    prepared_df = FrameHandler.prepend_ones(prepared_df)
    return prepared_df