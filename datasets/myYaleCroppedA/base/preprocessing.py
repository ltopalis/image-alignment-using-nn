
import os
import numpy as np
import h5py

try:
    from scipy.io import loadmat
except Exception:
    loadmat = None


def _is_hdf5_file(path: str) -> bool:
    try:
        with h5py.File(path, "r"):
            return True
    except Exception:
        return False


def _load_mat_legacy(path: str, mat_struct: str | None):
    if loadmat is None:
        raise RuntimeError(
            "scipy.io.loadmat δεν είναι διαθέσιμο. Εγκατάστησε scipy ή χρησιμοποίησε v7.3 MAT.")
    d = loadmat(path)
    d = {k: v for k, v in d.items() if not k.startswith("__")}
    if mat_struct is None:
        return d

    if mat_struct not in d:
        raise KeyError(
            f"Στο {path} δεν βρέθηκε struct '{mat_struct}'. Keys: {list(d.keys())[:30]}")
    s = d[mat_struct]
    if isinstance(s, np.ndarray) and s.shape == (1, 1):
        s = s[0, 0]
    if hasattr(s, "dtype") and s.dtype.names:
        out = {}
        for name in s.dtype.names:
            out[name] = s[name]
            if isinstance(out[name], np.ndarray) and out[name].shape == (1, 1):
                try:
                    out[name] = out[name][0, 0]
                except Exception:
                    pass
        return out
    if isinstance(s, dict):
        return s
    raise RuntimeError(
        f"Δεν μπόρεσα να κάνω parse το struct '{mat_struct}' από {path}.")


def _load_mat_v73_hdf5(path: str, mat_struct: str | None):

    f = h5py.File(path, "r")

    def list_datasets_under(group):
        out = {}

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = obj
        group.visititems(visitor)
        return out

    if mat_struct is None:

        out = {}
        for k in f.keys():
            if isinstance(f[k], h5py.Dataset):
                out[k] = f[k]
        return f, out

    if mat_struct not in f:
        f.close()
        raise KeyError(
            f"Στο {path} δεν βρέθηκε group/dataset '{mat_struct}'. Root keys: {list(f.keys())[:30]}")

    g = f[mat_struct]
    if isinstance(g, h5py.Dataset):

        return f, {mat_struct: g}

    ds_map = list_datasets_under(g)

    out = {}
    for rel_name, ds in ds_map.items():
        key = rel_name.split("/")[-1]

        if key in out:
            key = rel_name
        out[key] = ds
    return f, out


class MatSource:

    def __init__(self, path: str, mat_struct: str | None):
        self.path = path
        self.mat_struct = mat_struct
        self._is_v73 = _is_hdf5_file(path)
        self._h5_file = None
        self._map = None

        if self._is_v73:
            self._h5_file, self._map = _load_mat_v73_hdf5(path, mat_struct)
        else:
            self._map = _load_mat_legacy(path, mat_struct)

    def keys(self):
        return list(self._map.keys())

    def get_shape_dtype(self, key: str):
        obj = self._map[key]
        if isinstance(obj, h5py.Dataset):
            return obj.shape, obj.dtype
        arr = np.asarray(obj)

        arr = np.squeeze(arr)
        return arr.shape, arr.dtype

    def iter_slices_axis0(self, key: str, chunk_n: int = 128):

        obj = self._map[key]
        if isinstance(obj, h5py.Dataset):
            ds = obj
            if ds.ndim == 0:

                yield np.asarray(ds[()])[None]
                return
            n0 = ds.shape[0]
            for start in range(0, n0, chunk_n):
                end = min(n0, start + chunk_n)
                block = ds[start:end]
                block = np.asarray(block)
                block = np.squeeze(block)
                yield block
            return

        arr = np.asarray(obj)
        arr = np.squeeze(arr)
        if arr.ndim == 0:
            yield arr[None]
            return
        n0 = arr.shape[0]
        for start in range(0, n0, chunk_n):
            end = min(n0, start + chunk_n)
            yield arr[start:end]

    def close(self):
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass
            self._h5_file = None


def create_or_validate_output_datasets(out_h5: h5py.File, keys, first_shapes_dtypes):

    for k in keys:
        shape, dtype = first_shapes_dtypes[k]
        if len(shape) == 0:
            raise ValueError(
                f"Key '{k}' είναι scalar στο πρώτο αρχείο. Δεν υποστηρίζεται concat scalar.")

        init_shape = (0,) + tuple(shape[1:])
        max_shape = (None,) + tuple(shape[1:])
        chunks = True
        out_h5.create_dataset(k, shape=init_shape,
                              maxshape=max_shape, dtype=dtype, chunks=chunks)


def append_block(ds: h5py.Dataset, block: np.ndarray):
    block = np.asarray(block)
    block = np.squeeze(block)

    if block.ndim == ds.ndim - 1:
        block = block[None, ...]

    if block.ndim != ds.ndim:
        raise ValueError(
            f"Block ndim mismatch for '{ds.name}': got {block.shape}, expected ndims={ds.ndim}")

    if tuple(block.shape[1:]) != tuple(ds.shape[1:]):
        raise ValueError(
            f"Inner-shape mismatch for '{ds.name}': block {block.shape} vs dataset {ds.shape}"
        )

    n_add = block.shape[0]
    old_n = ds.shape[0]
    ds.resize((old_n + n_add,) + ds.shape[1:])
    ds[old_n:old_n + n_add] = block


def raw_concatenate(mat_paths, out_path, keys, mat_struct=None, chunk_n=128, verify=False):
    if len(mat_paths) == 0:
        raise ValueError("Δεν έδωσες καθόλου .mat αρχεία.")

    src0 = MatSource(mat_paths[0], mat_struct=mat_struct)

    for k in keys:
        if k not in src0._map:
            src0.close()
            raise KeyError(
                f"Στο πρώτο MAT δεν βρέθηκε key '{k}'. Διαθέσιμα: {src0.keys()}")

    first_shapes_dtypes = {k: src0.get_shape_dtype(k) for k in keys}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path):
        raise FileExistsError(
            f"Το output '{out_path}' υπάρχει ήδη. Διάλεξε άλλο όνομα ή διέγραψέ το.")

    with h5py.File(out_path, "w") as out_h5:

        out_h5.attrs["raw_concat_source"] = "mat_files"
        out_h5.attrs["mat_struct"] = "" if mat_struct is None else mat_struct
        out_h5.attrs["keys"] = json_dump_list(keys)
        out_h5.attrs["mat_files"] = json_dump_list(
            [os.path.abspath(p) for p in mat_paths])

        create_or_validate_output_datasets(out_h5, keys, first_shapes_dtypes)

        for mi, mp in enumerate(mat_paths):
            src = MatSource(mp, mat_struct=mat_struct)

            for k in keys:
                if k not in src._map:
                    src.close()
                    raise KeyError(
                        f"Στο '{mp}' λείπει key '{k}'. Διαθέσιμα: {src.keys()}")

                shp, dt = src.get_shape_dtype(k)
                shp0, dt0 = first_shapes_dtypes[k]

                if np.dtype(dt) != np.dtype(dt0):
                    src.close()
                    raise TypeError(
                        f"Key '{k}' dtype mismatch στο '{mp}': {dt} vs {dt0}")
                if tuple(shp[1:]) != tuple(shp0[1:]):
                    src.close()
                    raise ValueError(
                        f"Key '{k}' inner shape mismatch στο '{mp}': {shp} vs {shp0}")

            for k in keys:
                ds_out = out_h5[k]
                for block in src.iter_slices_axis0(k, chunk_n=chunk_n):
                    append_block(ds_out, block)

            src.close()

    src0.close()

    if verify:
        raw_verify(mat_paths, out_path, keys,
                   mat_struct=mat_struct, sample_per_file=3)


def raw_verify(mat_paths, out_path, keys, mat_struct=None, sample_per_file=3, seed=0):
    rng = np.random.default_rng(seed)
    print("\n[VERIFY] Raw sample checks...")

    with h5py.File(out_path, "r") as out_h5:
        offset = 0
        for mp in mat_paths:
            src = MatSource(mp, mat_struct=mat_struct)

            shp, _ = src.get_shape_dtype(keys[0])
            n = int(shp[0])

            idxs = rng.choice(n, size=min(sample_per_file, n), replace=False)
            idxs = sorted(int(i) for i in idxs)

            for k in keys:
                ds = out_h5[k]
                for i in idxs:

                    mat_block = next(_iter_single(src, k, i))
                    h5_block = ds[offset + i: offset + i + 1]
                    mat_arr = np.squeeze(np.asarray(mat_block))
                    h5_arr = np.squeeze(np.asarray(h5_block))

                    if not np.array_equal(mat_arr, h5_arr):
                        src.close()
                        raise AssertionError(
                            f"[VERIFY FAIL] {k}: '{mp}' idx={i} != out_h5 idx={offset+i}"
                        )
            print(
                f"[OK] {os.path.basename(mp)} matches raw (sampled {len(idxs)} indices).")
            offset += n
            src.close()

    print("[VERIFY] ✅ Passed.")


def _iter_single(src: MatSource, key: str, i: int):

    pos = 0
    for block in src.iter_slices_axis0(key, chunk_n=256):
        block = np.asarray(block)
        block = np.squeeze(block)
        if block.ndim == 0:

            yield block[None]
            return
        n0 = block.shape[0] if block.ndim >= 1 else 1
        if pos <= i < pos + n0:
            j = i - pos
            samp = block[j:j+1] if block.ndim >= 1 else block
            yield samp
            return
        pos += n0
    raise IndexError(f"Index {i} out of range for key '{key}' in {src.path}")


def json_dump_list(xs):

    return "[" + ",".join([repr(x) for x in xs]) + "]"


if __name__ == "__main__":
    from glob import glob

    raw_concatenate(
        mat_paths=glob("datasets/myYaleCroppedA/base/*.mat"),
        out_path="./datasets/myYaleCroppedA/base/dataset.hdf5",
        keys=['test_pts', 'template_affine', "m", "img", "tmplt", "p_init"],
        mat_struct="data",
        chunk_n=1000,
        verify="store_true",
    )
    print(f"\nDone. Wrote: datasets/myYaleCroppedA/base/dataset.hdf5")
