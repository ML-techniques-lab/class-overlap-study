from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import generate_base_datasets
import pandas as pd
import numpy as np
import random
import json
import os

# Parâmetros para geração dos datasets
# (Sobreposição/Overlap)
# class_sep=0.1 (muito sobreposto) até class_sep=2.0 (bem separado)
START_SEP = 0.1
END_SEP = 3
STEP_SEP = 0.3

# Nome das pastas para salvar os arquivos de saída
DATASETS_DIR = "datasets"
METADATAS_DIR = "metadatas"
PLOT_DIR = "plots"

# N_TESTS é 100 como definido em 'generate_base_datasets.py'
N_TESTS = 100

random.seed(generate_base_datasets.RANDOM_STATE)

def generate_datasets():
  # Criando diretorios caso não existam
  if not os.path.exists(DATASETS_DIR):
    os.mkdir(DATASETS_DIR)

  if not os.path.exists(METADATAS_DIR):
    os.mkdir(METADATAS_DIR)

 # 1. Gerar intervalo de valores para class_sep (Separação)
  sep_values = np.arange(START_SEP, END_SEP, STEP_SEP)

  parameters = generate_base_datasets.generate_parameters()
  for i in range(len(parameters)):
      dataset_name = f"dataset_{i+1}"
      
      if not os.path.exists(os.path.join(DATASETS_DIR, dataset_name)):
            os.mkdir(os.path.join(DATASETS_DIR, dataset_name))

      # Save base parameters used
      with open(os.path.join(METADATAS_DIR, f"metadata_dataset_{i+1}.json"), 'w') as json_file:
        json.dump(parameters[i], json_file, indent=2)

      # 2. Alteração principal: Iterar sobre 'sep_values'
      for sep in sep_values:
          # Define a separação atual (varia o overlap)
          parameters[i]["class_sep"] = sep
          
          # Fixa o balanceamento (50/50) para isolar o efeito do overlap
          parameters[i]["weights"] = [0.5] 

          X, y = make_classification(**parameters[i])
          dataset = pd.DataFrame(data=X)
          dataset['target'] = y
          
          # 3. Ajustar o nome do arquivo para refletir o 'sep'
          file_name = f"dataset_{i+1}_sep_{sep:.2f}.csv"
          dataset.to_csv(os.path.join(DATASETS_DIR, dataset_name, file_name), index=False)

  # Ajustar a chamada do plot
  plot_overlap_multiple_figures(sep_values)


def plot_overlap(sep_values):
  if not os.path.exists(PLOT_DIR):
    os.mkdir(PLOT_DIR)

  dataset_plots = []
  for i in range(4):
      dataset_levels = []
      for sep in sep_values:
          dataset_name = f"dataset_{i+1}"
          # Lendo o arquivo com o novo padrão de nome
          dataset_file = dataset_name + f"_sep_{sep:.2f}.csv"
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
    
    for j in range(len(sep_values)):
      X = dataset[j].iloc[:, :-1]
      y = dataset[j].iloc[:, -1]
      X_pca = pca.fit_transform(X)      
      current_sep = sep_values[j]       
      ax = axes[ax_idx]
      ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=10, cmap="viridis")
      ax.get_xaxis().set_visible(False)
      ax.set_yticks([])
      [spine.set_visible(False) for spine in ax.spines.values()]
      ax.set_title(f"class_sep={current_sep:.2f}", fontsize=6)
      ax_idx += 1

  plt.tight_layout()
  plt.savefig(os.path.join(PLOT_DIR, "overlap_plot.png"))

def plot_overlap_multiple_figures(sep_values, datasets_per_figure=10):
    if not os.path.exists(PLOT_DIR):
        os.mkdir(PLOT_DIR)

    N_SEP_LEVELS = len(sep_values)
    N_FIGURES = N_TESTS // datasets_per_figure

    for figure_idx in range(N_FIGURES):
        start_idx = figure_idx * datasets_per_figure
        end_idx = start_idx + datasets_per_figure
        
        # 1. Criar a figura (layout 10 linhas x N_SEP_LEVELS colunas)
        # O figsize é ajustado para manter a legibilidade
        fig, axes = plt.subplots(datasets_per_figure, N_SEP_LEVELS, figsize=(20, 1.8 * datasets_per_figure))
        axes = axes.flatten()
        ax_idx = 0

        # 2. Carregar e plotar os dados para o lote de datasets atual
        for i in range(start_idx, end_idx):
            dataset_num = i + 1
            
            for j, sep in enumerate(sep_values):
                dataset_name = f"dataset_{dataset_num}"
                dataset_file = dataset_name + f"_sep_{sep:.2f}.csv"
                
                # Carregar dataset
                try:
                    dataset = pd.read_csv(os.path.join(DATASETS_DIR, dataset_name, dataset_file))
                except FileNotFoundError:
                    # Continua se o arquivo não for encontrado (embora não deva acontecer se a geração for bem sucedida)
                    ax_idx += 1
                    continue
                    
                X = dataset.iloc[:, :-1]
                y = dataset.iloc[:, -1]
                
                # PCA e Plotagem
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                
                ax = axes[ax_idx]
                
                # Definir o rótulo da linha (apenas no primeiro subplot da linha)
                if j == 0:
                    ax.set_ylabel(f"Dataset {dataset_num}", fontsize=6)
                
                ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=10, cmap="viridis")
                ax.get_xaxis().set_visible(False)
                ax.set_yticks([])
                [spine.set_visible(False) for spine in ax.spines.values()]
                
                # Definir o título da coluna (apenas na primeira linha da figura)
                if i == start_idx:
                    ax.set_title(f"class_sep={sep:.2f}", fontsize=7)
                
                ax_idx += 1

        # 3. Salvar a figura com o índice do lote
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"overlap_plot_part_{figure_idx+1}.png"))

if __name__ == '__main__':
  generate_datasets()