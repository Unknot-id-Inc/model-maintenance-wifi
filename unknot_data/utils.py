from pathlib import Path

from gufu_data import _parse_db_record
from unknot_data.base import GeoLocationData
from unknot_data.timestamped_pose import TimestampedGeoPose, TimestampedPose
from unknot_data.vector import WifiBssidVector


def normalize_rssi(rssi: float | None) -> float:
    if rssi is None:
        return 0.0
    return 2 * (rssi + 100)


def scale_rssi(rssi: float | None) -> float:
    if rssi is None:
        return 0.0
    return (rssi + 120) / 120


def load_scan_from_txt(
    source_path: str, geo_reference: GeoLocationData
) -> tuple[list[WifiBssidVector], list[TimestampedGeoPose]]:
    text = Path(source_path).read_bytes().decode("utf-8")

    bssids = []
    poses = []

    for row_number, line in enumerate(text.splitlines(), start=1):
        xy, readings, _ = _parse_db_record(line)

        bssid = WifiBssidVector(
            timestamp=row_number,
            V=[normalize_rssi(x[1]) for x in readings],
            bssid=[x[0] for x in readings],
        )
        pose = TimestampedPose(
            timestamp=row_number,
            x=xy[0],
            y=xy[1],
            z=0.0,
            qw=1.0,
            qx=0.0,
            qy=0.0,
            qz=0.0,
        ).to_timestamped_geo_pose(geo_reference)

        bssids.append(bssid)
        poses.append(pose)
    return bssids, poses


# from pathlib import Path
# import numpy as np
# from gufu_data import _parse_db_record
# from unknot_data.base import GeoLocationData
# from unknot_data.timestamped_pose import TimestampedGeoPose, TimestampedPose
# from unknot_data.vector import WifiBssidVector, WifiRssiVector
# from unknot_data.wifi_networks import WifiNetwork
# from unknot_data.rssi_index import RssiIndex, RssiIntegerIndex
# from unknot_data.utils import normalize_rssi
# from unknot_data.utils import load_scan_from_txt
# from unknot_data.data_viz import AxisGrid


# positions = dict(zip(
#     aligned_reference_table.index.tolist(),
#     aligned_reference_table[['x', 'y']].values.tolist()
# ))

# rssis = {scan.scan_id: scan.rss_dbm for scan in updated_reference.scans}

# vocab = np.array(bssid_vocabulary)
# geo_reference = GeoLocationData(longitude=-74.6520576, latitude=40.3637847, altitude=0.0)

# bssids, poses = [], []

# for i, (scan_id, xy) in enumerate(positions.items()):
#     scan = np.array(rssis[scan_id])

#     bssid = WifiBssidVector(
#         timestamp=i,
#         V=[normalize_rssi(x) for x in scan[scan>-120].tolist()],
#         bssid=[x for x in vocab[scan>120].tolist()],
#     )
#     pose = TimestampedPose(
#         timestamp=i,
#         x=xy[0],
#         y=xy[1],
#         z=0.0,
#         qw=1.0,
#         qx=0.0,
#         qy=0.0,
#         qz=0.0,
#     ).to_timestamped_geo_pose(geo_reference)

#     bssids.append(bssid)
#     poses.append(pose)
