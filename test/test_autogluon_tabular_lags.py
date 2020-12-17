import pandas as pd
import numpy as np
from gluonts.nursery.autogluon_tabular_lags import LocalTabularPredictor
from gluonts.dataset.common import ListDataset
from gluonts.dataset.util import to_pandas


def test_autogluon_tabular():
    # create a dataset
    dataset = ListDataset(
        [
            {
                "start": pd.Timestamp("1750-01-04 00:00:00", freq="W-SUN"),
                "target": np.array(
                    [1089.2, 1078.91, 1099.88, 35790.55, 34096.95, 34906.95],
                ),
            },
            {
                "start": pd.Timestamp("1750-01-04 00:00:00", freq="W-SUN"),
                "target": np.array(
                    [1099.2, 1098.91, 1069.88, 35990.55, 34076.95, 34766.95],
                ),
            },
            {
                "start": pd.Timestamp("1750-01-04 00:00:00", freq="W-SUN"),
                "target": np.array(
                    [1079.2, 1078.91, 1029.88, 35790.55, 34056.95, 34566.95],
                ),
            },
            {
                "start": pd.Timestamp("1750-01-04 00:00:00", freq="W-SUN"),
                "target": np.array(
                    [2059.2, 1057.91, 1019.88, 35590.55, 34036.95, 34366.95],
                ),
            },
            {
                "start": pd.Timestamp("1750-01-04 00:00:00", freq="W-SUN"),
                "target": np.array(
                    [1089.2, 1078.91, 1099.88, 35790.55, 34056.95, 34566.95],
                ),
            },
            {
                "start": pd.Timestamp("1750-01-04 00:00:00", freq="W-SUN"),
                "target": np.array(
                    [1049.2, 1057.91, 1049.88, 35590.55, 34036.95, 34366.95],
                ),
            },
        ],
        freq="W-SUN",
    )
    prediction_length = 2
    context_length = 3
    freq = "W-SUN"
    predictor = LocalTabularPredictor(
        freq=freq, prediction_length=prediction_length, context_length=context_length
    )
    forecasts_it = predictor.predict(dataset)
    forecasts = list(forecasts_it)

    for entry, forecast in zip(dataset, forecasts):
        ts = to_pandas(entry)
        start_timestamp = ts.index[-1] + pd.tseries.frequencies.to_offset(freq)
        assert forecast.samples.shape[1] == prediction_length
        assert forecast.start_date == start_timestamp
    return forecasts


if __name__ == "__main__":
    print(test_autogluon_tabular())
