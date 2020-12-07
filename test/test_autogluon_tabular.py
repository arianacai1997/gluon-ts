import pandas as pd
import numpy as np
from gluonts.nursery.autogluon_tabular import LocalTabularPredictor
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import Evaluator


def test_autogluon_tabular(verbose=False):
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

    predictor = LocalTabularPredictor(freq="W-SUN", prediction_length=1,)
    forecasts = predictor.predict(dataset)

    # (optional) do evaluation
    if verbose:
        evaluator = Evaluator(quantiles=[0.5])
        tss = change_test(dataset, 10, "W-SUN")
        agg_metrics, item_metrics = evaluator(iter(tss), iter(list(forecasts)))
        print(agg_metrics)
    return list(forecasts)


def convert_df(data_iterator, freq):
    for data_entry in data_iterator:
        data = data_entry.copy()
        index = pd.date_range(
            start=data["start"], freq=freq, periods=data["target"].shape[-1],
        )
        data["ts"] = pd.DataFrame(index=index, data=data["target"].transpose())
        yield data


def change_test(dataset, prediction_length, freq):
    for data_entry in convert_df(iter(list(dataset)), freq):
        yield data_entry["ts"][-prediction_length:]


if __name__ == "__main__":
    print(test_autogluon_tabular())
