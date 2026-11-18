"""Request and response bodies. Pydantic checks every request before the model sees it."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from trip_duration import config


class TripRequest(BaseModel):
    """A trip as it is known at booking time: where from, where to, and when.

    No distance: the model looks up the typical distance between the two zones
    itself, so callers can't send a distance the model would misread.
    """

    # Unknown fields (such as an old client's trip_distance) get a 422, not silence.
    model_config = ConfigDict(extra="forbid")

    PULocationID: int = Field(
        ge=config.MIN_ZONE_ID,
        le=config.MAX_ZONE_ID,
        description="Pickup taxi zone (NYC TLC zone ID, 1-265).",
        examples=[161],
    )
    DOLocationID: int = Field(
        ge=config.MIN_ZONE_ID,
        le=config.MAX_ZONE_ID,
        description="Dropoff taxi zone (NYC TLC zone ID, 1-265).",
        examples=[236],
    )
    pickup_datetime: datetime = Field(
        description="Planned pickup time, local New York time.",
        examples=["2025-01-23T17:30:00"],
    )


class Prediction(BaseModel):
    """The predicted duration and which model version produced it."""

    predicted_duration_minutes: float = Field(examples=[18.7])
    model_name: str = Field(examples=["trip-duration"])
    model_version: str = Field(examples=["1"])


class BatchPrediction(BaseModel):
    """One prediction per trip, in the same order as the request."""

    predicted_duration_minutes: list[float] = Field(examples=[[18.7, 9.2]])
    model_name: str = Field(examples=["trip-duration"])
    model_version: str = Field(examples=["1"])


class Health(BaseModel):
    """Whether the API has a model loaded, and which one."""

    status: str = Field(examples=["ok"])
    model_name: str
    model_alias: str
    model_version: str | None = None
    detail: str | None = None
