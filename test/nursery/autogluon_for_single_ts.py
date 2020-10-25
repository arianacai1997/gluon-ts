from autogluon import TabularPrediction as task
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.util import to_pandas
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import pandas as pd
import matplotlib.pyplot as plt
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.forecast import SampleForecast
from gluonts.dataset.common import ListDataset
import json
import numpy as np
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class Gluonts3Auto:

    """
dataset_name:
Name of Gluonts common dataset
problem_type:
Autogluon task fit problem_type
num_samples:
Number of samples to draw on the model when evaluating.
"""

    def __init__(self, dataset_name, problem_type, num_samples, context_length=None):
        self.dataset = get_dataset(dataset_name, regenerate=False)
        self.problem_type = problem_type
        self.num_samples = num_samples
        self.context_length = context_length if context_length else self.dataset.metadata.prediction_length

    def augment_context(self):
        train = to_pandas(list(self.dataset.train)[0])
        return [None] * (self.context_length + self.dataset.metadata.prediction_length) + list(train.values)

    def parse_series(self, series):
        hour_of_day = series.index.hour
        month_of_year = series.index.month
        day_of_week = series.index.dayofweek
        year_idx = series.index.year
        target = series.values
        cal = calendar()
        holidays = cal.holidays(start=series.index.min(), end=series.index.max())
        df = pd.DataFrame(zip(year_idx, month_of_year, day_of_week, hour_of_day, series.index.isin(holidays), target),
                          columns=['year_idx', 'month_of_year', 'day_of_week', 'hour_of_day', 'holiday', 'target'])

        convert_type = {x: 'category' for x in df.columns.values[:4]}
        # cache = self.augment_context()
        # col = []
        # total = len(df)
        # for i in range(self.context_length):
        #     col.append('prev' + str(i))
        #     df['prev' + str(i)] = cache[i:i+total]
        df = df.astype(convert_type)
        return df

    def split_df(self, df):
        len = self.dataset.metadata.prediction_length
        return task.Dataset(df[:-2*len]), task.Dataset(df[-2*len:-len]), task.Dataset(df[-len:])

    def change_test(self):
        for data_entry in self.convert_df(iter([list(self.dataset.test)[0]])):
            yield data_entry["ts"]

    def convert_df(self, data_iterator):
        for data_entry in data_iterator:
            data = data_entry.copy()
            index = pd.date_range(
                start=data["start"],
                freq=self.dataset.metadata.freq,
                periods=data["target"].shape[-1],
            )
            data["ts"] = pd.DataFrame(
                index=index, data=data["target"].transpose()
            )
            yield data

    def to_sampleforecast(self, y_pred):
        l = 1
        freq = self.dataset.metadata.freq
        pred_len = self.dataset.metadata.prediction_length
        test = [list(self.dataset.test)[0]]
        forecasts = []
        for i in range(l):
            t = test[i]
            start = to_pandas(t).index[-pred_len]
            y_hat = np.array(y_pred[i * pred_len:(i + 1) * pred_len])
            samples = np.vstack([y_hat] * self.num_samples)
            # print('stacked samples:', samples)
            sample = SampleForecast(freq=freq, start_date=pd.Timestamp(start, freq=freq), item_id=str(i), samples=samples)
            forecasts.append(sample)
        return forecasts

    def evaluate_it(self, predictor, test_data):
        label_column = 'target'
        test_data_nolab = test_data.drop(labels=[label_column], axis=1)
        y_pred = predictor.predict(test_data_nolab)
        evaluator = Evaluator(quantiles=[0.5])
        # convert the test dataset
        ts_it = self.change_test()
        tss = list(ts_it)
        # convert the forecasts
        forecasts = self.to_sampleforecast(y_pred)  # y_pred
        agg_metrics, item_metrics = evaluator(iter([tss[0][-self.dataset.metadata.prediction_length:]]), iter(forecasts), num_series=1)
        print(y_pred)
        # print(tss.values[-5*self.dataset.metadata.prediction_length:])
        return json.dumps(agg_metrics, indent=4)

    def Wquantile(self, y, y_pred):
        assert len(y) == len(y_pred)
        res = 0
        for i in range(len(y)):
            res += abs(y.values[i][0]-y_pred[i])
        return res / sum(abs(y0) for y0 in y_pred)



g = Gluonts3Auto('m4_weekly', 'regression', 1, context_length=None)
ds = get_dataset('m4_weekly', regenerate=False)
test = list(ds.test)[0]
series = to_pandas(test)
df = g.parse_series(series)
task_tr, task_val, task_ts = g.split_df(df)
label_column = 'target'
problem_type = 'regression'
predictor = task.fit(train_data=task_tr, tuning_data=task_val, label=label_column, problem_type=problem_type,
                     output_directory='AutogluonModels/ag-a') #, presets='best_quality'
# print(predictor.fit_summary())
print(g.evaluate_it(predictor, task_ts))
