from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import generate_base_datasets
import seaborn as sns
import os

parameters = generate_base_datasets.generate_parameters()
datasets = generate_base_datasets.generate_datasets(parameters)

PLOT_DIR = "plots"

if os.path.exists(PLOT_DIR) == False:
    os.mkdir(PLOT_DIR)

# Comparação entre as escalas de cada feature de todos os datasets
fig, sub = plt.subplots(10, 10, figsize=(20, 20))

for i in range(len(datasets)):
  dataset = datasets[i][0]
  features_scale = []
  ax = sub[i // 10][i % 10]
  for feature in range(len(dataset[0])):
    scale = max(dataset[:,feature]) - min(dataset[:,feature])
    features_scale.append(scale)
  sns.barplot(features_scale, ax=ax)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_title(f"Dataset {i+1}\n\nn_features={parameters[i]['n_features']}", fontsize=6)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/feature_scales.png")


# PCA de cada dataset
fig, axes = plt.subplots(10, 10, figsize=(20, 20))
axes = axes.flatten()

# Iterar sobre os datasets e aplicar PCA
for i, dataset in enumerate(datasets):
    X, y = dataset
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Obter os parâmetros usados para criar o dataset
    args = parameters[i]
    params_label = (
        f"\nclusters={args['n_clusters_per_class']}, \n"
        f"class_sep={args['class_sep']:.2f}\n"
    )
    
    # Plotar os dados reduzidos
    ax = axes[i]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=10)
    ax.set_title(f'Dataset {i+1}\n{params_label}', fontsize=6)
    ax.axis('off')

# Ajustar o layout
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/pca_datasets.png")