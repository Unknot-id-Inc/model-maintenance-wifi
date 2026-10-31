import pickle

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from unknot_data.vector import WifiRssiVector


class WifiNetwork:
    def __init__(self, bssids: list[str]):
        self.bssids = bssids
        # Pre-compute lookup dict for O(1) index access
        self._bssid_to_idx = {b: i for i, b in enumerate(bssids)}

    def vectorize(
        self,
        bssids: list[str],
        rssis: list[int] | list[float],
        bssids_dict: dict[str, str] | None = None,
    ) -> list[int]:
        if bssids_dict is not None:
            bssids = [bssids_dict.get(b, b) for b in bssids]

        bssids, rssis = self.__filter_bssids(bssids, rssis)

        # Use cached dict for O(1) lookups instead of O(n) .index() calls
        idxs = [self._bssid_to_idx[b] for b in bssids]

        # Skip sparse matrix since output is always dense anyway
        v = np.zeros(len(self.bssids), dtype=np.float32)
        v[idxs] = rssis
        return v.astype(int).tolist()

    def __filter_bssids(self, bssids, rssis):
        new_bssids = []
        new_rssis = []

        # Use cached dict for O(1) membership check instead of O(n) list check
        for b, r in zip(bssids, rssis):
            if b in self._bssid_to_idx:
                new_bssids.append(b)
                new_rssis.append(r)
        return new_bssids, new_rssis

    @classmethod
    def from_pickle(cls, bssid_path):
        with open(bssid_path, "rb") as f:
            bssids_network = pickle.load(f)

        return cls(bssids=bssids_network)

    @classmethod
    def from_parquets(cls, parquet_paths: dict[str, str]):
        assert all([x in parquet_paths for x in ["bssids"]])

        bssids_array = WifiNetwork.parquet_to_numpy(parquet_paths["bssids"])
        wifi_network = cls(bssids=bssids_array.tolist())

        return wifi_network

    def to_parquets(self, parquet_paths: dict[str, str]):
        assert all([x in parquet_paths for x in ["bssids"]])

        bssids_table = self.numpy_to_table(np.array(self.bssids))
        pa.parquet.write_table(bssids_table, parquet_paths["bssids"])

    def bssid2rssi(self, bssids):
        return [WifiRssiVector.from_bssid_vector(b, self) for b in bssids]

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
    def parquet_to_numpy(parquet_file):
        # Step 1: Read the Parquet file into a PyArrow Table
        table = pq.read_table(parquet_file)

        # Step 2: Convert the PyArrow Table to a 2D numpy array
        # Stack each column (as a numpy array) horizontally to reconstruct the original 2D array
        array_2d = table.column(0).to_numpy()
        # array_2d = np.column_stack([table.column(i).to_numpy() for i in range(table.num_columns)])[:, 0].tolist()

        return array_2d
