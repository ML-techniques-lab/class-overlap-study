from sklearn.datasets import make_classification
from typing import Union
import numpy as np
import random

RANDOM_STATE = 10
N_SAMPLES = 1000
FLIP_Y = 0.01
N_TESTS = 100

random.seed(RANDOM_STATE)

Arguments = dict[str, Union[int, float, list[float]]]
Dataset = tuple[np.ndarray, np.ndarray]

def generate_parameters():
  parameters = []
  for i in range(N_TESTS):
    n_features = random.randint(10, 100)
    n_informative = random.randint(n_features // 2, n_features)
    n_redundant = random.randint((n_features-n_informative)//2, n_features-n_informative)
    n_clusters_per_class = random.randint(2, 5)
    class_sep = random.uniform(2, 3)
    scale = [random.randint(1, 100) for _ in range(n_features)]
    args = {
      'n_samples': N_SAMPLES,
      'n_features': n_features,
      'n_informative': n_informative,
      'n_redundant': n_redundant,
      'n_clusters_per_class': n_clusters_per_class,
      'class_sep': class_sep,
      'flip_y': FLIP_Y,
      'scale': scale,
      'random_state': RANDOM_STATE
    }
    parameters.append(args)
  return parameters

def generate_datasets(parameters):
  datasets = []
  for i in range(N_TESTS):
    dataset = make_classification(**parameters[i]) # type: ignore
    datasets.append(dataset)
  return datasets