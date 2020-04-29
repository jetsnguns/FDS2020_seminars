import re
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import dask.array as da


class AbstractPreprocessor(ABC):
    """
    Abstract class for all preprocessors. Operates on a single column. Works with Dask data frames.
    """
    def __init__(self, column=None):
        self.column = column

    @abstractmethod
    def apply(self, df):
        pass


class DropColumn(AbstractPreprocessor):
    """
    Removes a column by it's name.
    """
    def apply(self, df):
        return df.drop(self.column, axis=1)


class FillNAValue(AbstractPreprocessor):
    """
    Fill in missing values with the supplied constant.
    """
    def __init__(self, column=None, value=None):
        super().__init__(column=column)
        self.value = value

    def apply(self, df):
        df[self.column] = df[self.column].fillna(self.value)
        return df


class FillNAMethod(FillNAValue):
    """
    Impute missing values with the specified method. Assumes the method is implemented as a member of Dask Series.
    """
    def __init__(self, column=None, method='median'):
        super().__init__(column=column, value=None)
        self.method = method

    def apply(self, df):
        self.value = getattr(df[self.column], self.method)()
        return super().apply(df)


class FilterColumn(DropColumn):
    """
    Only keep row with the specified value in the specified column. Optionally remove the column after that.
    """
    def __init__(self, column=None, value=None, remove=True):
        super().__init__(column=column)
        self.value = value
        self.remove = remove

    def apply(self, df):
        df = df[df[self.column] == self.value]
        if self.remove:
            return super().apply(df)
        else:
            return df


class CyclicHM(AbstractPreprocessor):
    """
    Generate a pair of features for a time-based column.
    In this case it should be a float/int column of the format HHMM.
    """
    def apply(self, df):
        alpha = ((df[self.column] / 100) * 60 + (df[self.column] % 100)) / 1440
        df["cyclic_cos_" + self.column] = da.cos(2 * np.pi * alpha)
        df["cyclic_sin_" + self.column] = da.sin(2 * np.pi * alpha)

        return df


class SplitRegex(DropColumn):
    def __init__(self, column=None, re_str="", remove=True):
        super().__init__(column=column)
        self.re_str = re_str
        self.remove = remove

    def apply(self, df):
        df_new = df.map_partitions(lambda d: d[self.column].str.extract(self.re_str, expand=True))#.categorize()
        df = df.merge(df_new)
        if self.remove:
            return super().apply(df)
        else:
            return df


preprocessors_map = {
    "Drop": DropColumn,
    "FillMethod": FillNAMethod,
    "FillValue": FillNAValue,
    "Filter": FilterColumn,
    "CyclicHM": CyclicHM,
    "SplitRegex": SplitRegex
}


def create_preprocessor(desc):
    """
    Factory function to crete preprocessors from JSON descriptions. Assumes the name in property "name".
    :param desc:
    :return: desired preprocessor instance
    """
    kwargs = desc.copy()
    name = kwargs.pop("name")

    return preprocessors_map[name](**kwargs)
