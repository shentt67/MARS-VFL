import numpy as np

def float_parameter(level, maxval):
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def random_mask(data, level):
    level = float_parameter(sample_level(level), 0.5) + 0.1
    rate = np.random.random_sample(data.shape)
    drop_index = rate < level
    data = np.array(data)
    data[drop_index] = 0
    return data

def random_scale_up(data, level):
    level = float_parameter(sample_level(level), 0.5) + 0.1
    scale = 0.2
    rate = np.random.random_sample(data.shape)
    drop_index = rate < level
    data = np.array(data)
    data[drop_index] *= (scale + 1)
    return data

def random_scale_down(data, level):
    level = float_parameter(sample_level(level), 0.5) + 0.1
    scale = 0.8
    rate = np.random.random_sample(data.shape)
    drop_index = rate < level
    data = np.array(data)
    data[drop_index] *= scale
    return data

augmentations_general = [
    random_mask, random_scale_up, random_scale_down
]
