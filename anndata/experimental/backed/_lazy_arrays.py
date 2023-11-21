from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from xarray.core.indexing import (
    BasicIndexer,
    ExplicitlyIndexedNDArrayMixin,
    OuterIndexer,
)

from anndata._core.index import Index, _subset
from anndata._core.views import as_view
from anndata.compat import ZarrArray


class MaskedArrayMixIn(ExplicitlyIndexedNDArrayMixin):
    def __eq__(self, __o) -> np.ndarray:
        return self[...] == __o

    def __ne__(self, __o) -> np.ndarray:
        return ~(self == __o)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of this array

        Returns:
            Tuple[int, ...]: A shape that looks like a 1-d shape i.e., (#, )
        """
        return self.values.shape


class LazyCategoricalArray(MaskedArrayMixIn):
    __slots__ = (
        "values",
        "attrs",
        "_categories",
        "_categories_cache",
        "group",
        "_drop_unused_cats",
    )

    def __init__(self, codes, categories, attrs, drop_unused_cats, *args, **kwargs):
        """Class for lazily reading categorical data from formatted zarr group.   Used as base for `LazilyIndexedArray`.

        Args:
            codes (Union[zarr.Array, h5py.Dataset]): values (integers) of the array, one for each element
            categories (Union[zarr.Array, h5py.Dataset]): mappings from values to strings
            attrs (Union[zarr.Array, h5py.Dataset]): attrs containing boolean "ordered"
            _drop_unused_cats (bool): Whether or not to drop unused categories.
        """
        self.values = codes
        self._categories = categories
        self._categories_cache = None
        self.attrs = dict(attrs)
        self._drop_unused_cats = drop_unused_cats  # obsm/varm do not drop, but obs and var do.  TODO: Should fix in normal AnnData?

    @property
    def categories(self):  # __slots__ and cached_property are incompatible
        if self._categories_cache is None:
            if isinstance(self._categories, ZarrArray):
                self._categories_cache = self._categories[...]
            else:
                if (
                    "read_dataset" not in dir()
                ):  # avoid circular dependency, not sure what caused this all of a sudden after merging https://github.com/scverse/anndata/pull/949/commits/dc9f12fcbca977841e967c8414b9f1032e069250
                    from ..._io.h5ad import read_dataset
                self._categories_cache = read_dataset(self._categories)
        return self._categories_cache

    @property
    def dtype(self) -> pd.CategoricalDtype:
        return pd.CategoricalDtype(self.categories, self.ordered)

    @property
    def ordered(self):
        return bool(self.attrs["ordered"])

    def __getitem__(self, selection) -> pd.Categorical:
        idx = selection
        if isinstance(selection, BasicIndexer) or isinstance(selection, OuterIndexer):
            idx = selection.tuple[0]  # need to better understand this
        if isinstance(self.values, ZarrArray):
            codes = self.values.oindex[idx]
        else:
            codes = self.values[idx]
        if codes.shape == ():  # handle 0d case
            codes = np.array([codes])
        res = pd.Categorical.from_codes(
            codes=codes,
            categories=self.categories,
            ordered=self.ordered,
        )
        if self._drop_unused_cats:
            return res.remove_unused_categories()
        return res

    def __repr__(self) -> str:
        return f"LazyCategoricalArray(codes=..., categories={self.categories}, ordered={self.ordered})"

    def copy(self) -> LazyCategoricalArray:
        """Returns a copy of this array which can then be safely edited

        Returns:
            LazyCategoricalArray: copied LazyCategoricalArray
        """
        arr = LazyCategoricalArray(
            self.values, self._categories, self.attrs
        )  # self.categories reads in data
        return arr


class LazyMaskedArray(MaskedArrayMixIn):
    __slots__ = ("mask", "values", "_dtype_str")

    def __init__(self, values, mask, dtype_str, *args, **kwargs):
        """Class for lazily reading categorical data from formatted zarr group.  Used as base for `LazilyIndexedArray`.

        Args:
            values (Union[zarr.Array, h5py.Dataset]): Integer/Boolean array of values
            mask (Union[zarr.Array, h5py.Dataset]): mask indicating which values are non-null
            dtype_str (Nullable): one of `nullable-integer` or `nullable-boolean`
        """
        self.values = values
        self.mask = mask
        self._dtype_str = dtype_str

    @property
    def dtype(self) -> pd.CategoricalDtype:
        if self.mask is not None:
            if self._dtype_str == "nullable-integer":
                return pd.arrays.IntegerArray
            elif self._dtype_str == "nullable-boolean":
                return pd.arrays.BooleanArray
        return pd.array

    def __getitem__(self, selection) -> pd.Categorical:
        idx = selection
        if isinstance(selection, BasicIndexer) or isinstance(selection, OuterIndexer):
            idx = selection.tuple[0]  # need to understand this better
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        values = np.array(self.values[idx])
        if self.mask is not None:
            mask = np.array(self.mask[idx])
            if self._dtype_str == "nullable-integer":
                return pd.arrays.IntegerArray(values, mask=mask)
            elif self._dtype_str == "nullable-boolean":
                return pd.arrays.BooleanArray(values, mask=mask)
        return pd.array(values)

    def __repr__(self) -> str:
        if self._dtype_str == "nullable-integer":
            return "LazyNullableIntegerArray"
        elif self._dtype_str == "nullable-boolean":
            return "LazyNullableBooleanArray"

    def copy(self) -> LazyMaskedArray:
        """Returns a copy of this array which can then be safely edited

        Returns:
            LazyMaskedArray: copied LazyMaskedArray
        """
        arr = LazyMaskedArray(self.values, self.mask, self._dtype_str)
        return arr


@_subset.register(xr.DataArray)
def _subset_masked(a: xr.DataArray, subset_idx: Index):
    return a[subset_idx]


@as_view.register(xr.DataArray)
def _view_pd_boolean_array(a: xr.DataArray, view_args):
    return a
