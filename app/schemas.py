"""Request and response models of the CivicAI API.

The request model is the only place where input is checked. The saved model does NOT
validate anything: it would happily answer for a latitude of 140.7 or a missing borough.
Every rule below is written so that the error message names the field and says what is wrong.

The field names, types and required/optional status are the same as in
models/model_meta.json ("input_fields"); tests/test_api.py checks that they stay in sync.
"""

from datetime import datetime, timedelta
from typing import List, Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NYC_TZ = ZoneInfo("America/New_York")

# A generous box around New York City. Points outside it are rejected with 422.
LATITUDE_BOX = (40.3, 41.1)
LONGITUDE_BOX = (-74.4, -73.5)

EARLIEST_CREATED_AT = datetime(2020, 1, 1)       # the NYC 311 table starts in 2020
MAX_FUTURE = timedelta(days=1)                   # allows for clocks that are a little ahead
MAX_TEXT_LENGTH = 200

CREATED_AT_EXAMPLE = "2026-11-09T22:15:00"


class FieldValueError(ValueError):
    """A validation error that names the field it belongs to.

    Used for a rule that involves two fields (latitude and longitude), where the
    error must point at the field that is missing, not at the one that was sent.
    """

    def __init__(self, field, message):
        super().__init__(message)
        self.field = field


class PredictRequest(BaseModel):
    """One 311 service request, as the city's data records it.

    Unknown extra fields are REJECTED (422): a misspelt field name such as "borogh" would
    otherwise be ignored silently and the prediction would be made without that value.
    """

    model_config = ConfigDict(
        extra="forbid",                  # unknown fields return 422 and name the field
        str_strip_whitespace=True,       # "  BROOKLYN " is treated as "BROOKLYN"; "   " becomes empty and is rejected
        json_schema_extra={"examples": [{
            "complaint_type": "Noise - Residential",
            "descriptor": "Loud Music/Party",
            "borough": "BROOKLYN",
            "location_type": "Residential Building/House",
            "open_data_channel_type": "PHONE",
            "incident_zip": "11201",
            "latitude": 40.69,
            "longitude": -73.99,
            "created_at": CREATED_AT_EXAMPLE,
        }]},
    )

    # ------------------------------------------------------------ required
    complaint_type: str = Field(
        ..., min_length=1, max_length=MAX_TEXT_LENGTH,
        description="Kind of request, as in the NYC 311 data. Any case is accepted. "
                    "A type the model has not seen is accepted and treated as 'Other' (the response carries a warning).")
    borough: str = Field(
        ..., min_length=1, max_length=MAX_TEXT_LENGTH,
        description="BROOKLYN, BRONX, MANHATTAN, QUEENS or STATEN ISLAND (any case). Anything else is treated as unknown, with a warning.")
    open_data_channel_type: str = Field(
        ..., min_length=1, max_length=MAX_TEXT_LENGTH,
        description="How the request was filed: PHONE, ONLINE or MOBILE (any case). Another value is accepted, with a warning.")
    created_at: datetime = Field(
        ...,
        description="When the request was filed, ISO 8601 date and time, for example 2026-11-09T22:15:00. "
                    "A time zone is converted to New York time. The server derives the hour, weekday, month and weekend flag from it. "
                    "Must be between 2020-01-01 and one day from now.")

    # ------------------------------------------------------------ optional
    descriptor: Optional[str] = Field(
        None, min_length=1, max_length=MAX_TEXT_LENGTH,
        description="More detail about the complaint. Leave the field out (or send null) if unknown; an empty string is rejected.")
    location_type: Optional[str] = Field(
        None, min_length=1, max_length=MAX_TEXT_LENGTH,
        description="Kind of place, for example 'Residential Building/House'. Leave out if unknown.")
    incident_zip: Optional[str] = Field(
        None, min_length=1, max_length=MAX_TEXT_LENGTH,
        description="5-digit ZIP code, sent as text (for example \"11201\"). A ZIP outside New York is accepted and ignored, with a warning.")
    latitude: Optional[float] = Field(
        None, strict=True, allow_inf_nan=False,
        description=f"Number between {LATITUDE_BOX[0]} and {LATITUDE_BOX[1]} (New York). Send it together with longitude, or send neither.")
    longitude: Optional[float] = Field(
        None, strict=True, allow_inf_nan=False,
        description=f"Number between {LONGITUDE_BOX[0]} and {LONGITUDE_BOX[1]} (New York). Send it together with latitude, or send neither.")

    # ------------------------------------------------------------ rules
    @field_validator("created_at", mode="before")
    @classmethod
    def check_created_at(cls, value):
        if not isinstance(value, str):
            raise ValueError(f"created_at must be text in ISO 8601 format, for example {CREATED_AT_EXAMPLE}")
        text = value.strip()
        shown = text[:40]
        if text[-1:] in ("Z", "z"):
            text = text[:-1] + "+00:00"
        if len(text) < 11 or text[10] not in "Tt ":
            raise ValueError(f"created_at '{shown}' must contain both a date and a time, for example {CREATED_AT_EXAMPLE}")
        try:
            parsed = datetime.fromisoformat(text.replace("t", "T"))
        except ValueError:
            raise ValueError(f"created_at '{shown}' is not a valid ISO 8601 date and time "
                             f"(for example {CREATED_AT_EXAMPLE}); check the month, day and hour") from None
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(NYC_TZ).replace(tzinfo=None)      # all times are New York local time
        latest = datetime.now(NYC_TZ).replace(tzinfo=None) + MAX_FUTURE
        if parsed < EARLIEST_CREATED_AT:
            raise ValueError("created_at is before 2020-01-01; the NYC 311 data used for this model starts in 2020")
        if parsed > latest:
            raise ValueError("created_at is in the future; send the time the request was filed")
        return parsed

    @field_validator("incident_zip")
    @classmethod
    def check_zip(cls, value):
        if value is not None and not (len(value) == 5 and value.isascii() and value.isdigit()):
            raise ValueError("incident_zip must be exactly 5 digits sent as text, for example \"11201\"")
        return value

    @field_validator("latitude")
    @classmethod
    def check_latitude(cls, value):
        if value is None:
            return value
        if not -90 <= value <= 90:
            raise ValueError(f"latitude {value:g} is not a possible latitude: it must be between -90 and 90 "
                             f"(New York City is at about 40.7)")
        if not LATITUDE_BOX[0] <= value <= LATITUDE_BOX[1]:
            raise ValueError(f"latitude {value:g} is outside the New York City area this service covers "
                             f"(allowed: {LATITUDE_BOX[0]} to {LATITUDE_BOX[1]})")
        return value

    @field_validator("longitude")
    @classmethod
    def check_longitude(cls, value):
        if value is None:
            return value
        if not -180 <= value <= 180:
            raise ValueError(f"longitude {value:g} is not a possible longitude: it must be between -180 and 180 "
                             f"(New York City is at about -74.0)")
        if not LONGITUDE_BOX[0] <= value <= LONGITUDE_BOX[1]:
            raise ValueError(f"longitude {value:g} is outside the New York City area this service covers "
                             f"(allowed: {LONGITUDE_BOX[0]} to {LONGITUDE_BOX[1]})")
        return value

    @model_validator(mode="after")
    def coordinates_come_together(self):
        """A point needs both numbers. The error names the one that is missing."""
        if self.latitude is not None and self.longitude is None:
            raise FieldValueError("longitude", "longitude is missing: send latitude and longitude together, or neither")
        if self.longitude is not None and self.latitude is None:
            raise FieldValueError("latitude", "latitude is missing: send latitude and longitude together, or neither")
        return self


class PredictResponse(BaseModel):
    """The model's answer for one request."""

    predicted_class: int = Field(..., description="1 = resolved within 7 days, 0 = not resolved within 7 days. "
                                                  "Decided by the model's own cut-off (model.predict()).")
    label: str = Field(..., description="Text for predicted_class")
    probability_resolved_within_7d: float = Field(..., description="Probability of class 1, between 0 and 1")
    probability_not_resolved_within_7d: float = Field(..., description="Probability of class 0 (1 minus the line above)")
    decision_threshold: float = Field(..., description="The cut-off used: predicted_class is 1 when the probability of class 1 is at or above it")
    threshold_days: int = Field(..., description="The 7 in 'resolved within 7 days'")
    model_version: str
    warnings: List[str] = Field(default_factory=list,
                                description="Plain-language notes, for example when a value was not seen in training and is treated as 'Other'")


class FieldProblem(BaseModel):
    field: str
    message: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    errors: List[FieldProblem] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    versions_match_model_meta: Optional[bool] = None
    version_mismatches: List[str] = Field(default_factory=list)
    message: Optional[str] = None
