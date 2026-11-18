"""A distance a booking app can know: the typical distance between two zones."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config

_PAIR_FACTOR = 1000  # zone IDs are below 1000, so PU * 1000 + DO is a unique key


def _pair_key(pickup: pd.Series, dropoff: pd.Series) -> np.ndarray:
    return pickup.to_numpy(dtype=np.int64) * _PAIR_FACTOR + dropoff.to_numpy(
        dtype=np.int64
    )


class ZonePairDistance(TransformerMixin, BaseEstimator):
    """Replace the driven distance with the typical distance between the two zones.

    The distance a trip actually covered is only known after it ends, so a model
    that learns from it can't be used at booking time. This step learns, from the
    training trips only, the median distance for each pickup-dropoff zone pair.

    - fit: needs `trip_distance` (the driven distance of the training trips).
    - transform: ignores and drops any `trip_distance`, and adds
      `typical_distance`. A zone pair never seen in training falls back to the
      median for its pickup zone, then to the median of all trips.

    Lookups are stored as plain numpy arrays so skops can save them safely.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "ZonePairDistance":
        trips = pd.DataFrame(
            {
                "pair": _pair_key(X["PULocationID"], X["DOLocationID"]),
                "pickup": X["PULocationID"].to_numpy(dtype=np.int64),
                "distance": X["trip_distance"].to_numpy(dtype=float),
            }
        )
        pairs = trips.groupby("pair")["distance"].median()
        pickups = trips.groupby("pickup")["distance"].median()
        self.pair_keys_ = pairs.index.to_numpy(dtype=np.int64)
        self.pair_medians_ = pairs.to_numpy(dtype=float)
        self.pickup_keys_ = pickups.index.to_numpy(dtype=np.int64)
        self.pickup_medians_ = pickups.to_numpy(dtype=float)
        self.overall_median_ = float(trips["distance"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.drop(columns="trip_distance", errors="ignore").copy()
        by_pair = pd.Series(self.pair_medians_, index=self.pair_keys_)
        by_pickup = pd.Series(self.pickup_medians_, index=self.pickup_keys_)
        typical = by_pair.reindex(_pair_key(X["PULocationID"], X["DOLocationID"]))
        fallback = by_pickup.reindex(X["PULocationID"].to_numpy(dtype=np.int64))
        typical = typical.fillna(pd.Series(fallback.to_numpy(), index=typical.index))
        out[config.TYPICAL_DISTANCE] = typical.fillna(self.overall_median_).to_numpy()
        return out
