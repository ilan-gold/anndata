import copy

import dask.dataframe as dd
import dask.array as da

import dask
from dask.base import tokenize
from dask.dataframe.io.io import from_map

import pandas as pd, numpy as np, zarr

from anndata.experimental import read_elem, write_elem
from anndata._io.specs.methods import read_elem_partial


class AnnDataDFIOFunction(dd.io.utils.DataFrameIOFunction):
    def __init__(self, group: zarr.Group):
        self.index_col = group.attrs["_index"]
        self._columns = group.attrs["column-order"]
        self.group = group
        
    @property
    def columns(self) -> list[str]:
        return self._columns
    
    def project_columns(self, columns):
        """Return a new AnnDataDFIOFunction object with
        a sub-column projection.
        """
        if columns == self.columns:
            return self
        func = copy.deepcopy(self)
        func._columns = columns
        return func
    
    def __call__(self, parts: list[tuple[int, int]]) -> pd.DataFrame:
        """Parts is a tuple of chunk indices"""
        df = pd.DataFrame({
            k: read_elem_partial(self.group[k], indices=slice(parts[0], parts[1]))
            for k in [self.index_col] + self.columns
        })
        df.set_index(self.index_col, inplace=True)

        if df.index.name == "_index":
            df.index.name = None
    
        return df


def read_df_schema(group: zarr.Group) -> pd.DataFrame:
    """Return empty typed dataframe for anndata formated DF"""
    index_col = group.attrs["_index"]
    columns = group.attrs["column-order"] + [index_col]
    meta = {}
    for k in columns:
        encoding_type = group[k].attrs["encoding-type"]
        if encoding_type == "categorical":

            meta[k] = pd.CategoricalDtype(
                categories=group[k]["categories"][:],
                ordered=group[k].attrs["ordered"]
            )
        elif encoding_type == "nullable-integer":
            dt = str(group[k]["values"].dtype)
            dt = dt[0].upper() + dt[1:]  # hacky
            meta[k] = dt
        elif encoding_type == "nullable-boolean":
            meta[k] = "boolean"
        else:
            meta[k] = str(group[k].dtype)
    df = pd.DataFrame({k: pd.Series([], dtype=meta[k]) for k in columns})
    df.set_index(index_col, inplace=True)

    return df

def read_anndata_df(group: zarr.Group) -> dd.DataFrame:
    meta = read_df_schema(group)
    index_key = group.attrs["_index"]
    chunk_size = group[index_key].chunks[0]
    total_size = len(group[index_key])
    parts = list(zip(
        range(0, total_size, chunk_size),
        range(chunk_size, total_size + chunk_size, chunk_size)
    ))
    
    return from_map(
        AnnDataDFIOFunction(group), # DataFrameIOFunction
        parts, # iterable of arguments to AnnDataDFIOFunction, probably read_elem_partial
        meta=meta, # Empty dataframe with dtypes
        divisions=None, # iterable of points along index where partitions occur
        label="read-anndata",
        token=tokenize(group.path, group.store.path),
        enforce_metadata=False,
    )