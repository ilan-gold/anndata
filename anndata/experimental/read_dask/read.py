from collections.abc import MutableMapping
import imp
from pathlib import Path
from typing import Union

import zarr
import dask.array as da

from ..._core.anndata import AnnData
from ..._core.sparse_dataset import SparseDataset
from ...compat import (
    _clean_uns,
)
from ..._io.utils import (
    _read_legacy_raw,
)
from ..._io.specs.registry import IOSpec, read_elem
from ..._io.read import read_dispatched
from .zarr_dask_dataframe import read_anndata_df as read_dataframe

def read_zarr_dask(store: Union[str, Path, MutableMapping, zarr.Group]) -> AnnData:
    """\
    Read from a hierarchical Zarr array store.

    Parameters
    ----------
    store
        The filename, a :class:`~typing.MutableMapping`, or a Zarr storage class.
    """
    if isinstance(store, Path):
        store = str(store)

    f = zarr.open_consolidated(store, mode="r")
    # Backwards compat
    def dispatch_element(read_func, group, k, iospec):
        if k in ["obs", "var"]:
            return read_dataframe(group[k])
        if k in ["varm", "obsm"]:
            return {key: da.from_zarr(val) for key, val in group[k].items()} 
        if iospec == IOSpec("csr_matrix", "0.1.0") or iospec == IOSpec("csc_matrix", "0.1.0"):
            mtx = SparseDataset(group[k]).to_backed()
            mtx.data = da.from_zarr(mtx.data)
            mtx.indices = da.from_zarr(mtx.indices)
            return mtx
        if k == "raw":
            return None
        return read_func(group[k])

    def dispatch_anndata_args(group, args):
        args["raw"] = _read_legacy_raw(
            group, args.get("raw"), read_dataframe, read_elem
        )

        if "X" in args:
            args["dtype"] = args["X"].dtype

        # Backwards compat to <0.7
        if isinstance(group["obs"], zarr.Array):
            _clean_uns(args)
        args["parse_df"] = False
        return args

    return read_dispatched(f, dispatch_element, dispatch_anndata_args)