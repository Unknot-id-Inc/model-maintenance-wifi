from typing import TYPE_CHECKING

import numpy as np
from pydantic import AliasChoices, Field

from unknot_data.base import TimestampedVector, alias_generator

if TYPE_CHECKING:
    from unknot_data.vector import WifiNetwork


class WifiBssidVector(TimestampedVector, populate_by_name=True, extra="forbid"):
    """
    WiFiAPData

    A class representing Wi-Fi RSSI vectors.

    Attributes:
        timestamp (int): The timestamp at which the data was collected.
        rssi (int): The RSSI value vector
        layer (str): The layer name
        idx (int): The index of the wifi rssi vector

    """

    bssid: list[str] = Field(
        [],
        validation_alias=AliasChoices(*alias_generator("bssid")),
        serialization_alias="bssid",
    )

    layer: str | None = Field(
        None,
        validation_alias=AliasChoices(*alias_generator("layer")),
        serialization_alias="layer",
    )

    idx: int | None = Field(
        None,
        validation_alias=AliasChoices(*alias_generator("idx")),
        serialization_alias="idx",
    )

    systime: int | None = Field(
        None,
        validation_alias=AliasChoices(*alias_generator("systime")),
        serialization_alias="systime",
    )

    scanid: str | None = Field(
        None,
        validation_alias=AliasChoices(*alias_generator("scanid")),
        serialization_alias="scanid",
    )

    ssid: list[str] | None = Field(
        None,
        validation_alias=AliasChoices(*alias_generator("ssid")),
        serialization_alias="ssid",
    )

    channel: list[int | float | None] | None = Field(
        None,
        validation_alias=AliasChoices(*alias_generator("Channel")),
        serialization_alias="Channel",
    )

    channel_width: list[int | float | None] | None = Field(
        None,
        validation_alias=AliasChoices(*alias_generator("ChannelWidth")),
        serialization_alias="ChannelWidth",
    )

    center_freq0: list[int | float | None] | None = Field(
        None,
        validation_alias=AliasChoices(*alias_generator("CenterFreq0")),
        serialization_alias="CenterFreq0",
    )

    center_freq1: list[int | float | None] | None = Field(
        None,
        validation_alias=AliasChoices(*alias_generator("CenterFreq1")),
        serialization_alias="CenterFreq1",
    )

    band: list[float | None] | None = Field(
        None,
        validation_alias=AliasChoices(*alias_generator("Band")),
        serialization_alias="Band",
    )

    systimes: list[int | None] | None = Field(
        None,
        validation_alias=AliasChoices(*alias_generator("systimes")),
        serialization_alias="systimes",
    )

    scanids: list[str | None] | None = Field(
        None,
        validation_alias=AliasChoices(*alias_generator("scanids")),
        serialization_alias="scanids",
    )
    timestamps: list[int | None] | None = Field(
        None,
        validation_alias=AliasChoices(*alias_generator("timestamps")),
        serialization_alias="timestamps",
    )
    updated: bool = Field(
        False,
        validation_alias=AliasChoices(*alias_generator("updated")),
        serialization_alias="updated",
    )

    @property
    def rssi(self):
        return self.V

    def asdict(self):
        return {
            "timestamp": self.timestamp,
            "systime": self.systime,
            "bssid": self.bssid,
            "rssi": self.V,
            "layer": self.layer,
            "ssid": self.ssid,
            "channel": self.channel,
            "channel_width": self.channel_width,
            "center_freq0": self.center_freq0,
            "center_freq1": self.center_freq1,
            "band": self.band,
        }

    def get_info(self, bssid: str):
        if bssid not in self.bssid:
            raise KeyError(f"{bssid} not found in bssid list {self.bssid[:5]}...")

        idx_bssid = self.bssid.index(bssid)
        rssi_value = self.rssi[idx_bssid]
        ssid = None
        channel = None
        channel_width = None
        center_freq0 = None
        center_freq1 = None
        band = None

        if (
            hasattr(self, "ssid")
            and isinstance(self.ssid, list)
            and len(self.ssid) == len(self.bssid)
        ):
            ssid = self.ssid[idx_bssid]
        if (
            hasattr(self, "channel")
            and isinstance(self.channel, list)
            and len(self.channel) == len(self.bssid)
        ):
            channel = self.channel[idx_bssid]
        if (
            hasattr(self, "channel_width")
            and isinstance(self.channel_width, list)
            and len(self.channel_width) == len(self.bssid)
        ):
            channel_width = self.channel_width[idx_bssid]
        if (
            hasattr(self, "center_freq0")
            and isinstance(self.center_freq0, list)
            and len(self.center_freq0) == len(self.bssid)
        ):
            center_freq0 = self.center_freq0[idx_bssid]
        if (
            hasattr(self, "center_freq1")
            and isinstance(self.center_freq1, list)
            and len(self.center_freq1) == len(self.bssid)
        ):
            center_freq1 = self.center_freq1[idx_bssid]
        if (
            hasattr(self, "band")
            and isinstance(self.band, list)
            and len(self.band) == len(self.bssid)
        ):
            band = self.band[idx_bssid]

        return {
            "timestamp": self.timestamp,
            "layer": self.layer,
            "bssid": bssid,
            "rssi_value": rssi_value,
            "ssid": ssid,
            "channel": channel,
            "channel_width": channel_width,
            "center_freq0": center_freq0,
            "center_freq1": center_freq1,
            "band": band,
        }


class WifiRssiVector(TimestampedVector, populate_by_name=True, extra="forbid"):
    """
    WiFiAPData

    A class representing Wi-Fi RSSI vectors.

    Attributes:
        timestamp (int): The timestamp at which the data was collected.
        rssi (int): The RSSI value vector
        layer (str): The layer name
        idx (int): The index of the wifi rssi vector

    """

    layer: str | None = Field(None)
    idx: int | None = Field(None)
    systime: int | None = Field(None)
    scanid: str | None = Field(None)

    systimes: list[int] | list[None] | None = Field(
        None,
        validation_alias=AliasChoices(*alias_generator("systimes")),
        serialization_alias="systimes",
    )
    scanids: list[str] | list[None] | None = Field(
        None,
        validation_alias=AliasChoices(*alias_generator("scanids")),
        serialization_alias="scanids",
    )
    timestamps: list[int] | list[None] | None = Field(
        None,
        validation_alias=AliasChoices(*alias_generator("timestamps")),
        serialization_alias="timestamps",
    )

    @property
    def rssi(self):
        return self.V

    def asdict(self):
        return {
            "timestamp": self.timestamp,
            "V": self.V,
            "layer": self.layer,
            "idx": self.idx,
            "systime": self.systime,
            "systimes": self.systimes,
            "scanids": self.scanids,
            "timestamps": self.timestamps,
        }

    @classmethod
    def from_bssid_vector(
        cls, bssid_vector: WifiBssidVector, wifi_network: "WifiNetwork"
    ):
        rssi_vector = WifiRssiVector(
            timestamp=int(bssid_vector.timestamp),
            V=wifi_network.vectorize(
                bssids=bssid_vector.bssid, rssis=bssid_vector.rssi
            ),
            layer=bssid_vector.layer,
            idx=bssid_vector.idx,
            systime=bssid_vector.systime,
            scanid=bssid_vector.scanid,
        )

        return rssi_vector

    def arr(self):
        return np.array(self.V)
