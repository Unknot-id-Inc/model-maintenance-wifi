import numpy as np
import pymap3d as pm

from unknot_data.base import GeoLocationData, TimestampedData


class TimestampedPose(TimestampedData):
    x: float
    y: float
    z: float
    qw: float
    qx: float
    qy: float
    qz: float
    batch_id: int | None = None
    pose_id: int | None = None
    layer: str | None = None
    x_cov: float | None = None
    y_cov: float | None = None
    z_cov: float | None = None
    zone: str | None = None
    location_id: str | int | None = None

    @classmethod
    def origin(cls, timestamp: int = 0) -> "TimestampedPose":
        pose = TimestampedPose(
            timestamp=timestamp,
            x=0.0,
            y=0.0,
            z=0.0,
            qx=0.0,
            qy=0.0,
            qz=0.0,
            qw=1.0,
            zone=None,
        )
        return pose

    @property
    def position(self):
        return (self.x, self.y, self.z)

    @property
    def orientation(self):
        return (self.qw, self.qx, self.qy, self.qz)

    def to_timestamped_geo_pose(
        self, geo_location_reference: GeoLocationData, in_place: bool = False
    ) -> "TimestampedGeoPose":
        latitude, longitude, altitude = pm.enu2geodetic(
            self.x,
            self.y,
            self.z,
            geo_location_reference.latitude,
            geo_location_reference.longitude,
            geo_location_reference.altitude,
        )

        timestamped_pose = TimestampedGeoPose(
            timestamp=self.timestamp,
            x=self.x,
            y=self.y,
            z=self.z,
            qx=self.qx,
            qy=self.qy,
            qz=self.qz,
            qw=self.qw,
            batch_id=self.batch_id,
            pose_id=self.pose_id,
            layer=self.layer,
            x_cov=self.x_cov,
            y_cov=self.y_cov,
            z_cov=self.z_cov,
            longitude=longitude,
            latitude=latitude,
            altitude=altitude,
            location_id=self.location_id,
        )

        if in_place:
            self = timestamped_pose
        else:
            return timestamped_pose

    def deepcopy(self) -> "TimestampedPose":
        return TimestampedPose(
            timestamp=self.timestamp,
            x=self.x,
            y=self.y,
            z=self.z,
            qx=self.qx,
            qy=self.qy,
            qz=self.qz,
            qw=self.qw,
            batch_id=self.batch_id,
            pose_id=self.pose_id,
            layer=self.layer,
            x_cov=self.x_cov,
            y_cov=self.y_cov,
            z_cov=self.z_cov,
            zone=self.zone,
            location_id=self.location_id,
        )

    def asdict(self) -> dict[str, int | float | str]:
        return {
            "timestamp": self.timestamp,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "qx": self.qx,
            "qy": self.qy,
            "qz": self.qz,
            "qw": self.qw,
            "batch_id": self.batch_id,
            "pose_id": self.pose_id,
            "layer": self.layer,
            "x_cov": self.x_cov,
            "y_cov": self.y_cov,
            "z_cov": self.z_cov,
            "zone": self.zone,
            "location_id": self.location_id,
        }


class TimestampedGeoPose(TimestampedPose):
    longitude: float
    latitude: float
    altitude: float | None = None

    @classmethod
    def origin(
        cls, timestamp: int, geo_location_reference: GeoLocationData
    ) -> "TimestampedGeoPose":
        geo_pose = TimestampedPose(
            timestamp=timestamp,
            x=0.0,
            y=0.0,
            z=0.0,
            qx=0.0,
            qy=0.0,
            qz=0.0,
            qw=1.0,
        ).to_timestamped_geo_pose(geo_location_reference)
        return geo_pose

    @classmethod
    def from_xy(
        cls,
        timestamp: int,
        x: float,
        y: float,
        geo_location_reference: GeoLocationData,
        z: float = 0.0,
    ) -> "TimestampedGeoPose":
        geo_pose = TimestampedPose(
            timestamp=timestamp,
            x=x,
            y=y,
            z=z,
            qx=0.0,
            qy=0.0,
            qz=0.0,
            qw=1.0,
        ).to_timestamped_geo_pose(geo_location_reference)
        geo_pose.altitude = None
        return geo_pose

    def recompute_geo(
        self, geo_location_reference: GeoLocationData
    ) -> "TimestampedGeoPose":
        latitude, longitude, altitude = pm.enu2geodetic(
            self.x,
            self.y,
            self.z,
            geo_location_reference.latitude,
            geo_location_reference.longitude,
            geo_location_reference.altitude,
        )

        self.longitude = longitude
        self.latitude = latitude
        self.altitude = altitude

        return TimestampedGeoPose(
            timestamp=self.timestamp,
            x=self.x,
            y=self.y,
            z=self.z,
            qx=self.qx,
            qy=self.qy,
            qz=self.qz,
            qw=self.qw,
            batch_id=self.batch_id,
            pose_id=self.pose_id,
            layer=self.layer,
            x_cov=self.x_cov,
            y_cov=self.y_cov,
            z_cov=self.z_cov,
            zone=self.zone,
            location_id=self.location_id,
            longitude=longitude,
            latitude=latitude,
            altitude=altitude,
        )

    def recompute_xyz(self, geo_location_reference: GeoLocationData):
        e, n, u = pm.geodetic2enu(
            lat=self.latitude,
            lon=self.longitude,
            h=self.altitude,
            lat0=geo_location_reference.latitude,
            lon0=geo_location_reference.longitude,
            h0=geo_location_reference.altitude,
        )
        return TimestampedGeoPose(
            timestamp=self.timestamp,
            x=e,
            y=n,
            z=u,
            qx=self.qx,
            qy=self.qy,
            qz=self.qz,
            qw=self.qw,
            batch_id=self.batch_id,
            pose_id=self.pose_id,
            layer=self.layer,
            x_cov=self.x_cov,
            y_cov=self.y_cov,
            z_cov=self.z_cov,
            zone=self.zone,
            location_id=self.location_id,
            longitude=self.longitude,
            latitude=self.latitude,
            altitude=self.altitude,
        )

    def deepcopy(self) -> "TimestampedGeoPose":
        return TimestampedGeoPose(
            timestamp=self.timestamp,
            x=self.x,
            y=self.y,
            z=self.z,
            qx=self.qx,
            qy=self.qy,
            qz=self.qz,
            qw=self.qw,
            batch_id=self.batch_id,
            pose_id=self.pose_id,
            layer=self.layer,
            x_cov=self.x_cov,
            y_cov=self.y_cov,
            z_cov=self.z_cov,
            zone=self.zone,
            location_id=self.location_id,
            longitude=self.longitude,
            latitude=self.latitude,
            altitude=self.altitude,
        )

    def asdict(self) -> dict[str, int | float | str]:
        return {
            "timestamp": self.timestamp,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "qx": self.qx,
            "qy": self.qy,
            "qz": self.qz,
            "qw": self.qw,
            "batch_id": self.batch_id,
            "pose_id": self.pose_id,
            "layer": self.layer,
            "x_cov": self.x_cov,
            "y_cov": self.y_cov,
            "z_cov": self.z_cov,
            "zone": self.zone,
            "location_id": self.location_id,
            "longitude": self.longitude,
            "latitude": self.latitude,
            "altitude": self.altitude,
        }

    def arr(self):
        return np.array((self.x, self.y))
