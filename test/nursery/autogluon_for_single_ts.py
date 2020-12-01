from gluonts.model.predictor import Predictor
# Standard library imports
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterator,
    Iterable, Dict,
)

# Third-party imports
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import pandas as pd
from autogluon import TabularPrediction as task

# First-party imports
from gluonts.dataset.common import DataEntry, Dataset, ListDataset
from gluonts.dataset.util import to_pandas
from gluonts.model.forecast import Forecast, SampleForecast
from gluonts.model.predictor import Localizer
from gluonts.model.estimator import Estimator

if TYPE_CHECKING:  # avoid circular import
    from gluonts.model.estimator import Estimator  # noqa

OutputTransform = Callable[[DataEntry, np.ndarray], np.ndarray]


class TabularPredictor(Predictor):
    def __init__(self,
                 ag_model,
                 freq: str,
                 prediction_length: int,
                 ) -> None:
        self.ag_model = ag_model  # task?
        self.freq = freq
        self.prediction_length = prediction_length

    def predict(self, dataset: Iterable[Dict], lag=0, model=None, as_pandas=False) -> Iterator[SampleForecast]:
        def get_prediction_dataframe(series, lag=0, model=None, as_pandas=False):
            hour_of_day = series.index.hour
            month_of_year = series.index.month
            day_of_week = series.index.dayofweek
            year_idx = series.index.year
            target = series.values
            cal = calendar()
            holidays = cal.holidays(start=series.index.min(), end=series.index.max())
            df = pd.DataFrame(
                zip(year_idx, month_of_year, day_of_week, hour_of_day, series.index.isin(holidays), target),
                columns=['year_idx', 'month_of_year', 'day_of_week', 'hour_of_day', 'holiday', 'target'])
            convert_type = {x: 'category' for x in df.columns.values[:4]}
            df = df.astype(convert_type)
            return df

        for idx in range(len(list(dataset))):
            ts = to_pandas(list(dataset)[idx])
            df = get_prediction_dataframe(ts, lag)
            output = self.ag_model.predict(df)  # type: dataframe
            yield self.to_forecast(dataset, output, idx)

    def to_forecast(self, gluonts_test, y_pred, i):
        test = [list(gluonts_test)[i]]
        forecasts = []
        t = test[i]
        start = to_pandas(t).index[-self.prediction_length]
        y_hat = np.array(y_pred[i * self.prediction_length:(i + 1) * self.prediction_length])
        samples = np.vstack([y_hat])  # * num_samples
        sample = SampleForecast(freq=self.freq, start_date=pd.Timestamp(start, freq=self.freq), item_id=str(i),
                                samples=samples)
        forecasts.append(sample)
        return forecasts


class TabularEstimator(Estimator):
    def __init__(self, freq, prediction_length):
        super().__init__()
        self.task = task
        self.freq = freq
        self.prediction_length = prediction_length

    def train(self, training_data: Dataset) -> TabularPredictor:
        def get_prediction_dataframe(series, lag=0, model=None, as_pandas=False):
            hour_of_day = series.index.hour
            month_of_year = series.index.month
            day_of_week = series.index.dayofweek
            year_idx = series.index.year
            target = series.values
            cal = calendar()
            holidays = cal.holidays(start=series.index.min(), end=series.index.max())
            df = pd.DataFrame(
                zip(year_idx, month_of_year, day_of_week, hour_of_day, series.index.isin(holidays), target),
                columns=['year_idx', 'month_of_year', 'day_of_week', 'hour_of_day', 'holiday', 'target'])
            convert_type = {x: 'category' for x in df.columns.values[:4]}
            df = df.astype(convert_type)
            return df

        df = get_prediction_dataframe(to_pandas(list(training_data)[0])[:-self.prediction_length])
        ag_model = self.task.fit(df, label="target")
        return TabularPredictor(ag_model, self.freq, self.prediction_length)


def LocalTabularPredictor(*args, **kwargs):
    return Localizer(TabularEstimator(*args, **kwargs))


def stack_ts(dataset):
    series = []
    for i, entry in enumerate(list(dataset.test)):
        series.append(to_pandas(entry))
    return series


if __name__ == '__main__':
    from gluonts.dataset.repository.datasets import get_dataset

    dataset = get_dataset("m4_weekly", regenerate=False)
    series_ls = stack_ts(dataset)
    predictor = LocalTabularPredictor(freq=dataset.metadata.freq,
                                      prediction_length=dataset.metadata.prediction_length)
    forecasts = predictor.predict(list(dataset.test))
