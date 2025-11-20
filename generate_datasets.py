from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import generate_base_datasets
import pandas as pd
import numpy as np
import random
import json
import os

# Taxa de desbalanceamento inicial e final e o passo entre as taxas
START = 1
END = 141
STEP = 15.5

# Nome das pastas para salvar os arquivos de saída
DATASETS_DIR = "datasets"
METADATAS_DIR = "metadatas"
PLOT_DIR = "plots"

random.seed(generate_base_datasets.RANDOM_STATE)

def generate_datasets():
  # Criando diretorios caso não existam
  if not os.path.exists(DATASETS_DIR):
    os.mkdir(DATASETS_DIR)

  if not os.path.exists(METADATAS_DIR):
    os.mkdir(METADATAS_DIR)

  ratios = np.arange(START, END, STEP)
  weights = list(map(lambda ratio: 1/(ratio + 1), ratios))

  # generate datasets
  parameters = generate_base_datasets.generate_parameters()
  for i in range(len(parameters)):
    dataset_name = f"dataset_{i+1}"

    if not os.path.exists(os.path.join(DATASETS_DIR, dataset_name)):
      os.mkdir(os.path.join(DATASETS_DIR, dataset_name))

    # Save base parameters used
    with open(os.path.join(METADATAS_DIR, f"metadata_dataset_{i+1}.json"), 'w') as json_file:
      json.dump(parameters[i], json_file, indent=2)

    # Create datasets for each weight 
    for weight in weights:
      parameters[i]["weights"] = [weight]
      X, y = make_classification(**parameters[i])
      dataset = pd.DataFrame(data=X)
      dataset['target'] = y
      dataset.to_csv(os.path.join(DATASETS_DIR, dataset_name, f"dataset_{i+1}_w_{weight:.3f}.csv"), index=False)
  
  plot_imbalance_ratios(weights)


def plot_imbalance_ratios(weights):
  if not os.path.exists(PLOT_DIR):
    os.mkdir(PLOT_DIR)

  dataset_plots = []

  for i in range(4):
    dataset_levels = []
    for w in weights:
      dataset_name = f"dataset_{i+1}"
      dataset_file = dataset_name + f"_w_{w:.3f}.csv"
      dataset = pd.read_csv(os.path.join(DATASETS_DIR, dataset_name, dataset_file))
      dataset_levels.append(dataset)
    dataset_plots.append(dataset_levels)

  # Renderizando plot
  _, axes = plt.subplots(4, 10, figsize=(20, 6))
  axes = axes.flatten()
  ax_idx = 0
  pca = PCA(n_components=2)

  for i in range(len(dataset_plots)):
    dataset = dataset_plots[i]
    axes[ax_idx].set_ylabel(f"Dataset {i+1}", fontsize=6)
    for j in range(len(weights)):
      X = dataset[j].iloc[:, :-1]
      y = dataset[j].iloc[:, -1]
      X_pca = pca.fit_transform(X)
      weight = weights[j]
      ir = (1/weight) - 1
      ax = axes[ax_idx]
      ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=10, cmap="viridis")
      ax.get_xaxis().set_visible(False)
      ax.set_yticks([])
      [spine.set_visible(False) for spine in ax.spines.values()]
      ax.set_title(f"weight={weight:.3f}\nIR={ir:.3f}", fontsize=6)
      ax_idx += 1

  plt.tight_layout()
  plt.savefig(os.path.join(PLOT_DIR, "imbalance_plot.png"))

if __name__ == '__main__':
  generate_datasets()