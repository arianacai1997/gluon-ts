# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.


from itertools import islice

import matplotlib.pyplot as plt

from gluonts.dataset.util import to_pandas
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.nursery.autogluon_tabular import TabularEstimator


def run_example():
    dataset = get_dataset("electricity")

    estimator = TabularEstimator(
        freq="H",
        prediction_length=24,
        time_limits=2 * 60,  # two minutes
    )

    n_train = 5

    training_data = list(islice(dataset.train, n_train))

    predictor = estimator.train(
        training_data=training_data,
    )

    forecasts = list(predictor.predict(training_data))

    for entry, forecast in zip(training_data, forecasts):
        ts = to_pandas(entry)
        plt.figure()
        plt.plot(ts[-7 * predictor.prediction_length :], label="target")
        forecast.plot()
        plt.show()


if __name__ == "__main__":
    run_example()
