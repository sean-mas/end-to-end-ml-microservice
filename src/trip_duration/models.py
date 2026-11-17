"""Model recipes. Each one is a full sklearn Pipeline that takes build_features output.

Because every pipeline picks its own columns, the API can always send the same
feature table, whichever model is in production.
"""

import pickle

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from . import config, features

# Starting point for the forest. sklearn's defaults (100 trees, leaves of 1 trip)
# would need tens of GB for 2 million trips, so leaves hold at least 50 trips.
RF_START = {"n_estimators": 50, "max_depth": 20, "min_samples_leaf": 50}


def _pick(columns: list[str]) -> ColumnTransformer:
    """Keep only the given columns, in order."""
    return ColumnTransformer([("keep", "passthrough", columns)], remainder="drop")


def mean_baseline() -> Pipeline:
    """Always predict the average training duration. Any model must beat this."""
    return Pipeline([("pick", _pick(config.FEATURES)), ("model", DummyRegressor())])


def linear_route() -> Pipeline:
    """The course's reference model: one-hot pickup_dropoff route + distance."""
    encode = ColumnTransformer(
        [
            ("route", OneHotEncoder(handle_unknown="ignore"), [config.ROUTE_FEATURE]),
            ("distance", "passthrough", ["trip_distance"]),
        ]
    )
    return Pipeline(
        [
            ("add_route", FunctionTransformer(features.add_route)),
            ("encode", encode),
            ("model", LinearRegression()),
        ]
    )


def random_forest(columns: list[str] = config.FEATURES, **params) -> Pipeline:
    """A Random Forest on the given feature columns."""
    settings = {**RF_START, **params}
    forest = RandomForestRegressor(
        n_jobs=-1, random_state=config.RANDOM_STATE, **settings
    )
    return Pipeline([("pick", _pick(columns)), ("model", forest)])


def size_mb(model: Pipeline) -> float:
    """Return how many MB the fitted model takes when saved."""
    estimator = model.named_steps["model"]
    if isinstance(estimator, RandomForestRegressor):
        # Counting the tree arrays is exact enough and avoids copying a large model.
        state = (tree.tree_.__getstate__() for tree in estimator.estimators_)
        return sum(s["nodes"].nbytes + s["values"].nbytes for s in state) / 1e6
    return len(pickle.dumps(model)) / 1e6
