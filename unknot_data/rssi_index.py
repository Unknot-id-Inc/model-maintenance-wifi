import pickle
from collections import Counter

import hnswlib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
from scipy import sparse

from unknot_data.vector import WifiBssidVector, WifiRssiVector


def dense_to_sparse(
    dense_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sparse_array = sp.csr_matrix(dense_array)

    data = sparse_array.data
    row_indices = sparse_array.indices
    col_pointers = sparse_array.indptr

    return data, row_indices, col_pointers


def sparse_to_dense(sparse: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    dense_array = sp.csr_matrix(sparse).todense()

    return dense_array


import json
from datetime import datetime

import numpy as np


def save_arrays_to_parquet_fsl(
    ints: np.ndarray, floats: np.ndarray, strings: np.ndarray, path: str
) -> None:
    """
    Save three arrays into a single Parquet file using FixedSizeList:

      - ints:    shape (N, k),  dtype int16
      - floats:  shape (N, 2),  dtype float32
      - strings: shape (N,),    string-like

    Stored as:
      - 'int_features'  : FixedSizeList<int16>[k]
      - 'float_coords'  : FixedSizeList<float32>[2]
      - 'labels'        : string
    """

    # Normalize inputs
    ints = np.asarray(ints)
    floats = np.asarray(floats)
    strings = np.asarray(strings)

    # Shape checks
    if ints.ndim != 2:
        raise ValueError(f"ints must be 2D (N, k), got shape {ints.shape}")
    if floats.ndim != 2 or floats.shape[1] != 2:
        raise ValueError(f"floats must be shape (N, 2), got shape {floats.shape}")
    if strings.ndim != 1:
        raise ValueError(f"strings must be 1D (N,), got shape {strings.shape}")

    N, k = ints.shape
    if floats.shape[0] != N or strings.shape[0] != N:
        raise ValueError("ints, floats, and strings must have the same length N")

    # Enforce dtypes
    ints = ints.astype(np.int16, copy=False)
    floats = floats.astype(np.float32, copy=False)
    strings = strings.astype(str)

    # ---- Build FixedSizeList for ints ----
    # Flatten ints to a 1D buffer, then wrap as FixedSizeListArray
    int_values = pa.array(ints.reshape(-1), type=pa.int16())
    int_list = pa.FixedSizeListArray.from_arrays(int_values, list_size=k)

    # ---- Build FixedSizeList for floats ----
    float_values = pa.array(floats.reshape(-1), type=pa.float32())
    float_list = pa.FixedSizeListArray.from_arrays(float_values, list_size=2)

    # ---- Labels ----
    label_array = pa.array(strings, type=pa.string())

    table = pa.table(
        {
            "int_features": int_list,
            "float_coords": float_list,
            "labels": label_array,
        }
    )

    # Metadata: store k in case you ever change list_size logic
    meta = {
        "int_features_k": k,
        "float_coords_dim": 2,
    }
    metadata = {b"array_info": json.dumps(meta).encode("utf-8")}
    table = table.replace_schema_metadata(metadata)

    pq.write_table(table, path)


def load_arrays_from_parquet_fsl(path: str):
    """
    Load arrays saved by save_arrays_to_parquet_fsl.

    Returns:
      ints    : np.ndarray of shape (N, k), dtype int16
      floats  : np.ndarray of shape (N, 2), dtype float32
      strings : np.ndarray of shape (N,),   dtype str
    """
    table = pq.read_table(path)

    # --- Read metadata (optional but nice sanity check) ---
    md = table.schema.metadata or {}
    info = {}
    if b"array_info" in md:
        info = json.loads(md[b"array_info"].decode("utf-8"))

    # ---------- int_features ----------
    int_chunked = table["int_features"]
    int_array = int_chunked.combine_chunks()  # FixedSizeListArray

    if not isinstance(int_array.type, pa.FixedSizeListType):
        raise TypeError("Column 'int_features' is not a FixedSizeListType")

    k = int_array.type.list_size
    # If metadata exists, cross-check
    if "int_features_k" in info and info["int_features_k"] != k:
        raise ValueError(f"Metadata says k={info['int_features_k']}, but list_size={k}")

    N = len(int_array)

    # Child buffer: flattened int16 values of length N * k
    int_values = int_array.values  # pa.Array
    ints_flat = int_values.to_numpy().astype(np.int16, copy=False)
    ints = ints_flat.reshape(N, k)

    # ---------- float_coords ----------
    float_chunked = table["float_coords"]
    float_array = float_chunked.combine_chunks()

    if not isinstance(float_array.type, pa.FixedSizeListType):
        raise TypeError("Column 'float_coords' is not a FixedSizeListType")

    d = float_array.type.list_size
    if d != 2:
        raise ValueError(f"Expected float_coords size 2, got {d}")
    if "float_coords_dim" in info and info["float_coords_dim"] != d:
        raise ValueError(
            f"Metadata says float_dim={info['float_coords_dim']}, but list_size={d}"
        )

    float_values = float_array.values  # pa.Array, length N * 2
    floats_flat = float_values.to_numpy().astype(np.float32, copy=False)
    floats = floats_flat.reshape(N, d)

    # ---------- labels ----------
    label_col = table["labels"]
    strings = np.asarray(label_col.to_pylist(), dtype=str)

    return ints, floats, strings


class RssiIndex:
    def __init__(
        self,
        positions: np.ndarray,
        rssis: np.ndarray,
        dim=525,
        space="cosine",
        ef_construction: int = 10000,
        M: int = 80,
        set_ef: int = 10000,
        build_index: bool = True,
        layers=None,
    ):
        self.positions = np.ascontiguousarray(positions.astype(np.float32, copy=False))
        self.rssis = np.ascontiguousarray(rssis.astype(np.float32, copy=False))
        self.layers = layers
        self.index = hnswlib.Index(space=space, dim=dim)

        if build_index:
            self.index.init_index(
                max_elements=len(self.rssis), ef_construction=ef_construction, M=M
            )
            self.index.add_items(self.rssis, np.array(list(range(len(rssis)))))
            self.index.set_ef(set_ef)

    def knn_query(
        self,
        rssi: np.ndarray,
        k: int,
        skip_label=None,
        return_distances=False,
        return_labels=False,
        return_layers=False,
    ):
        if skip_label is not None:
            labels, distances = self.index.knn_query(
                np.array(rssi).astype("float32"), k=k + 1
            )
        else:
            labels, distances = self.index.knn_query(
                np.array(rssi).astype("float32"), k=k
            )

        labels = labels[0]

        if skip_label is not None:
            labels = np.array([x for x in labels if x != skip_label])

        output = (self.rssis[labels][:k], self.positions[labels][:k])

        if return_distances:
            output += (distances[:k],)
        if return_labels:
            output += (labels[:k],)
        if return_layers:
            layers = [self.layers[i] for i in labels]
            output += (layers[:k],)

        return output

    def to_pickle(self, pickle_path):
        sparse_rssis = sparse.csr_matrix(self.rssis)

        with open(pickle_path, "wb") as f:
            pickle.dump((self.positions, sparse_rssis, self.index, self.layers), f)

    @classmethod
    def from_pickle(cls, pickle_path):
        with open(pickle_path, "rb") as f:
            positions, sparse_rssis, index, layers = pickle.load(f)
            rssis = sparse_rssis.todense()

        rssi_index = cls(positions, rssis, build_index=False)
        rssi_index.index = index
        rssi_index.layers = layers
        return rssi_index

    @classmethod
    def from_parquets(cls, index_dim: int, parquet_paths: dict[str, str], set_ef=10000):
        assert all(
            [
                x in parquet_paths
                for x in ["positions", "sparse_rssis", "index", "layers"]
            ]
        )

        start_time = datetime.now()

        positions_array = RssiIndex.parquet_to_numpy(
            parquet_paths["positions"], dtype=np.float32
        )
        rssis_array = RssiIndex.parquet_to_numpy(
            parquet_paths["sparse_rssis"], dtype=np.float32
        )
        layers_array = RssiIndex.parquet_to_numpy(parquet_paths["layers"])[:, 0].astype(
            "<U30"
        )

        print(
            f"loaded arrays in {(datetime.now() - start_time).total_seconds()} seconds"
        )
        start_time = datetime.now()

        # Load the index from the file
        index = hnswlib.Index(
            space="cosine", dim=index_dim
        )  # Replace 'l2' and 128 with the appropriate values
        index.load_index(parquet_paths["index"])

        print(
            f"loaded index in {(datetime.now() - start_time).total_seconds()} seconds"
        )

        rssi_index = cls(positions_array, rssis_array, build_index=False)

        rssi_index.index = index
        rssi_index.layers = layers_array
        return rssi_index

    def to_parquets(self, parquet_paths: dict[str, str]):
        assert all(
            [
                x in parquet_paths
                for x in ["positions", "sparse_rssis", "index", "layers"]
            ]
        )

        positions_table = self.numpy_to_table(self.positions)
        rssis_table = self.numpy_to_table(np.array(self.rssis))
        layers_table = self.numpy_to_table(self.layers)

        pa.parquet.write_table(positions_table, parquet_paths["positions"])
        pa.parquet.write_table(rssis_table, parquet_paths["sparse_rssis"])
        self.index.save_index(parquet_paths["index"])
        pa.parquet.write_table(layers_table, parquet_paths["layers"])

    @staticmethod
    def numpy_to_table(array: np.ndarray):
        if len(array.shape) == 2:
            table = pa.Table.from_arrays(
                [pa.array(array[:, i]) for i in range(array.shape[1])],
                names=[f"col_{i}" for i in range(array.shape[1])],
            )
        elif len(array.shape) == 1:
            table = pa.Table.from_arrays([pa.array(array)], names=["col_0"])
        return table

    @staticmethod
    def parquet_to_numpy(parquet_file, dtype=None):
        # Step 1: Read the Parquet file into a PyArrow Table
        table = pq.read_table(parquet_file)

        # Step 2: Convert the PyArrow Table to a 2D numpy array
        # Stack each column (as a numpy array) horizontally to reconstruct the original 2D array
        if dtype is None:
            array_2d = np.column_stack(
                [table.column(i).to_numpy() for i in range(table.num_columns)]
            )
        else:
            array_2d = np.empty(
                (table.num_rows, table.num_columns),
                dtype=dtype,
                order="C",
            )
            for index in range(table.num_columns):
                array_2d[:, index] = table.column(index).to_numpy(zero_copy_only=False)

        return array_2d

    def predict(self, rssi, k=15):
        return self.knn_query(rssi=rssi, k=k)[1].mean(axis=0)


from wifisfm.data.utils.dailoc_index_maker import (
    generate_cluster_dict,
    generate_cluster_layer_dict,
    generate_index_from_cluster_dicts,
)


class RssiIntegerIndex:
    def __init__(
        self,
        positions: np.ndarray,
        rssis: np.ndarray,
        layers: np.ndarray,
        dim: int,
        space="l2",
        ef_construction: int = 10000,
        M: int = 80,
        set_ef: int = 10000,
        build_index: bool = True,
    ):
        self.positions = positions.astype(np.float32)
        self.rssis = rssis  # .astype(np.float32)
        self.layers = layers
        self.index = hnswlib.Index(space=space, dim=dim)

        if build_index:
            self.index.init_index(
                max_elements=len(self.rssis), ef_construction=ef_construction, M=M
            )
            self.index.add_items(self.rssis, np.array(list(range(len(rssis)))))
            self.index.set_ef(set_ef)

    def knn_query(
        self,
        rssi: np.ndarray,
        k: int,
        skip_label=None,
        return_distances=False,
        return_labels=False,
        return_layers=False,
    ):
        if skip_label is not None:
            labels, distances = self.index.knn_query(
                np.array(rssi).astype("float64"), k=k + 1
            )
        else:
            labels, distances = self.index.knn_query(
                np.array(rssi).astype("float64"), k=k
            )

        labels = labels[0]

        if skip_label is not None:
            labels = np.array([x for x in labels if x != skip_label])

        output = (self.rssis[labels][:k], self.positions[labels][:k])

        if return_distances:
            output += (distances[:k],)
        if return_labels:
            output += (labels[:k],)
        if return_layers:
            layers = [self.layers[i] for i in labels]
            output += (layers[:k],)

        return output

    @classmethod
    def from_wifi_dataset(cls, wifi_dataset):
        positions, rssis_ints, layers = [], [], []

        for w in wifi_dataset:
            positions.extend([[p.x, p.y] for p in w.poses])
            rssis_ints.extend([p.rssi for p in w.rssis])
            layers.extend([p.layer for p in w.rssis])

        positions = np.array(positions)
        rssis_ints = np.array(rssis_ints)
        layers = np.array(layers)

        return cls(
            positions=positions,
            rssis=rssis_ints,
            layers=layers,
            dim=rssis_ints.shape[1],
            space="cosine",
        )

    @classmethod
    def from_wifi_dataset_clusters(
        cls, wifi_dataset, device_model_labels, hierarchical_struct
    ):
        cluster_dict = {}

        cluster_dict = generate_cluster_dict(
            wifi_dataset, device_model_labels, cluster_dict_partial=cluster_dict
        )

        cluster_layer_dict = generate_cluster_layer_dict(hierarchical_struct)

        index = generate_index_from_cluster_dicts(
            cluster_dict, cluster_layer_dict, hierarchical_struct
        )

        return index

    @classmethod
    def from_wifi_datasets(
        cls, wifi_dataset_paths, device_model_labels, hierarchical_struct
    ):
        cluster_dict = {}

        for wifi_data_path in wifi_dataset_paths:
            with open(wifi_data_path, "rb") as f:
                wifi_dataset = pickle.load(f)

            cluster_dict = generate_cluster_dict(
                wifi_dataset, device_model_labels, cluster_dict_partial=cluster_dict
            )

        cluster_layer_dict = generate_cluster_layer_dict(hierarchical_struct)

        index = generate_index_from_cluster_dicts(
            cluster_dict, cluster_layer_dict, hierarchical_struct
        )

        return index

    @classmethod
    def load_index(cls, wifi_config_dir, space="l2", set_ef=10000):
        parquet_path = f"{wifi_config_dir}/index.parquet"
        bin_path = f"{wifi_config_dir}/index.bin"

        start_time = datetime.now()
        ints, floats, strings = load_arrays_from_parquet_fsl(parquet_path)
        index_dim = ints.shape[1]
        print(
            f"loaded arrays in {(datetime.now() - start_time).total_seconds()} seconds"
        )
        start_time = datetime.now()

        index_bin = hnswlib.Index(space=space, dim=index_dim)
        index_bin.load_index(bin_path)
        print(
            f"loaded index in {(datetime.now() - start_time).total_seconds()} seconds"
        )

        index = cls(
            positions=floats,
            rssis=ints,
            layers=strings,
            dim=index_dim,
            build_index=False,
        )
        index.index = index_bin
        index.index.set_ef(set_ef)
        return index

    def save_index(self, wifi_config_dir):
        parquet_path = f"{wifi_config_dir}/index.parquet"
        bin_path = f"{wifi_config_dir}/index.bin"

        save_arrays_to_parquet_fsl(
            ints=self.rssis.astype(np.int32),
            floats=self.positions,
            strings=self.layers,
            path=parquet_path,
        )

        self.index.save_index(bin_path)

    def compute_index_ratio_score(
        self,
        rssi_vector: np.ndarray,
        k: int = 50,
        distance_threshold=1.0,
        n_labels_min=2,
    ) -> dict[str, float]:
        rssi_vector = np.array(rssi_vector)
        query_layers, query_labels = self.query_layers_with_distance_threshold(
            rssi_vector, k=k, distance_threshold=distance_threshold
        )

        if len(query_labels) < n_labels_min:
            return {}

        index_ratio_score = self.initialize_index_ratio_scores()

        sample_counts = {layer: count for layer, count in Counter(query_layers).items()}
        # self.print_message("SAMPLE COUNTS {}".format(sample_counts), -1)
        unk_count = k - len(query_layers)
        class_weights = self.compute_class_weights(query_labels=query_labels)
        weighted_sample_counts = self.compute_weighted_sample_counts(
            sample_counts, unk_count, class_weights
        )
        # self.print_message(
        #     "WEIGHTED SAMPLE COUNTS {}".format(weighted_sample_counts), -1
        # )

        normalized_sample_counts = self.normalize_sample_counts(weighted_sample_counts)

        # self.print_message(
        #     "NORMALIZED SAMPLE COUNTS {}".format(normalized_sample_counts), -1
        # )

        for k, v in normalized_sample_counts.items():
            index_ratio_score[k] = v

        return index_ratio_score

    def normalize_sample_counts(
        self, weighted_sample_counts: dict[str, float]
    ) -> dict[str, float]:
        n_total = np.sum(list(weighted_sample_counts.values()))
        normalized_sample_counts = {
            k: v / n_total for k, v in weighted_sample_counts.items()
        }

        # print(normalized_sample_counts)
        return normalized_sample_counts

    def compute_weighted_sample_counts(
        self,
        sample_counts: dict[str, int],
        unk_count: float,
        class_weights: dict[str, float],
    ) -> dict[str, float]:
        weighted_sample_counts = {
            k: v * class_weights[k]
            for k, v in sample_counts.items()
            if k in class_weights
        }
        weighted_unk_count = np.sum(list(weighted_sample_counts.values())) * (
            unk_count / np.sum(list(sample_counts.values()))
        )
        weighted_sample_counts["UNK"] = weighted_unk_count

        # print(weighted_sample_counts)
        return weighted_sample_counts

    def compute_class_weights(self, query_labels: np.ndarray) -> dict[str, float]:
        xy_query = self.positions[query_labels]
        xy_mean = np.mean(xy_query, axis=0)

        d_from_mean = np.linalg.norm(self.positions - xy_mean, axis=-1)
        d_max = np.linalg.norm(xy_query - xy_mean, axis=-1).max()
        # print(d_max)

        class_counts = Counter(self.layers[d_from_mean < d_max])
        class_weights = {
            cls: len(self.layers) / count for cls, count in class_counts.items()
        }
        return class_weights

    def query_layers_with_distance_threshold(
        self, rssi_vector: np.ndarray, k: int, distance_threshold=1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        _, _, distances, labels, layers = self.knn_query(
            rssi=rssi_vector,
            k=k,
            return_distances=True,
            return_labels=True,
            return_layers=True,
        )

        distances = np.array(distances[0])
        layers = np.array(layers)

        # self.print_message("IRS LAYERS={}".format(layers), -1)
        # self.print_message("IRS  DISTS={}".format(distances), -1)

        layers = layers[distances < distance_threshold]
        labels = labels[distances < distance_threshold]

        return layers, labels

    def compute_min_rssi_distance_threshold(self, k: int = 50) -> float:
        max_dists = []

        for rssi_vector in self.rssis:
            _, _, distances, labels, layers = self.knn_query(
                rssi=rssi_vector,
                k=k,
                return_distances=True,
                return_labels=True,
                return_layers=True,
            )

            max_dists.append(distances[0][-1])

        min_rssi_distance = np.mean(max_dists) + 2 * np.std(max_dists)
        return min_rssi_distance

    def convert_bssidvector_to_rssivector(
        self, bssid_vector: WifiBssidVector
    ) -> WifiRssiVector:
        rssi_vector_unreduced = self.wifi_network.vectorize(
            bssids=bssid_vector.bssid, rssis=bssid_vector.rssi
        )
        rssi_vector_reduced = self.pca.reduce(rssi_vector_unreduced)[0].tolist()

        rssi_vector = WifiRssiVector(
            timestamp=bssid_vector.timestamp,
            V=rssi_vector_reduced,
        )
        return rssi_vector

    def initialize_index_ratio_scores(self):
        index_ratio_scores = {
            layer: 0.0 for layer in sorted(list(set(self.layers.tolist())))
        }
        index_ratio_scores["UNK"] = 0.0
        return index_ratio_scores
