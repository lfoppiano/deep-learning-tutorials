import os

data_dir = 'jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

lines = []

with open(fname, 'r') as f:
    lines = f.readlines()

header = lines[0].split(",")
data = lines[1:]

print(header)
data_size = len(data)
print(data_size)

import numpy as np

float_data = np.zeros((data_size, len(header) - 1))

for i, line in enumerate(data):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i] = values

# Normalisation
mean = float_data[:200000].mean(axis=0)
train_data = float_data[:200000] - mean
std = float_data[:200000].std(axis=0)
train_data = train_data / std


def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    """
        - lookback: how many timestaps back the input data should go
        - delay: how many timesteps in the future the target should be
        - min_index / max_index: min and max indexes from the input data to consider
        - shuffle: shuffle or not
        - step: period where you sample the data
        - batch_size: the number of sampels per batch
    """

    if max_index is None:
        max_index = len(data) - delay - 1

    # I need to advance by a lookback number of timestamps to avoid going out of bound
    i = min_index + lookback

    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))

        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


lookback = 1440
step = 6
delay = 144
batch_size = 128

training_generator = generator(float_data, lookback, delay, 0, 200000, True, step=2, batch_size=2)
validation_generator = generator(float_data, lookback, delay, 200001, 300000)
test_generator = generator(float_data, lookback, delay, 300001, None)


def evaluate_naive_method():
    batch_maes = []

    for step_ in range(step):
        samples, targets = next(validation_generator)

        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        # batch.maes


evaluate_naive_method()
