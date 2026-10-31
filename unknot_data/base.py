from pydantic import AliasChoices, BaseModel, Field
from pydantic.alias_generators import to_camel, to_pascal, to_snake


def alias_generator(field_name: str) -> list[str]:
    snake = to_snake(field_name)
    aliases = list(
        {field_name, field_name.lower(), to_camel(snake), snake, to_pascal(field_name)}
    )
    return aliases


class TimestampedData(BaseModel, extra="forbid", strict=True, populate_by_name=True):
    timestamp: int | float | None = Field(
        None,
        validation_alias=AliasChoices(*alias_generator("Timestamp")),
        serialization_alias="Timestamp",
    )


class TimestampedVector(TimestampedData, extra="forbid", strict=True):
    V: list[int | float] | None = Field(
        None,
        validation_alias=AliasChoices(*alias_generator("Vector")),
        serialization_alias="Vector",
    )


class GeoLocationData(BaseModel, extra="forbid", strict=True, populate_by_name=True):
    longitude: float = Field(
        validation_alias=AliasChoices(*alias_generator("Longitude")),
        serialization_alias="Longitude",
    )
    latitude: float = Field(
        validation_alias=AliasChoices(*alias_generator("Latitude")),
        serialization_alias="Latitude",
    )
    altitude: float = Field(
        0.0,
        validation_alias=AliasChoices(*alias_generator("Altitude")),
        serialization_alias="Altitude",
    )

    def asdict(self):
        return {
            "longitude": self.longitude,
            "latitude": self.latitude,
            "altitude": self.altitude,
        }

    def as_tuple(self, without_altitude: bool = False) -> tuple[float, float]:
        if without_altitude:
            return (self.latitude, self.longitude)
        else:
            return (self.latitude, self.longitude, self.altitude)
