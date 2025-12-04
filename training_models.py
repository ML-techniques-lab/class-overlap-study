import generate_datasets # Dependência para obter a lista de pesos (weights)
import pandas as pd
import random
import os
import sys
from multiprocessing import Pool, cpu_count
from sklearn.base import clone

# Métricas e model selection
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from imblearn.metrics import geometric_mean_score

# Modelos
from sklearn.neighbors import KNeighborsClassifier
from sklearn_lvq import GlvqModel
from typing import Dict, Any, List, Tuple

# --- Constantes e Configurações ---
SCALING_METHODS = ['original', 'MM', 'SS']
DATASETS_DIR = "scaled_datasets"
RANDOM_STATE = 10
N_FOLDS = 5

random.seed(RANDOM_STATE)

# Modelos a serem rodados (Usamos classes de modelos e parâmetros)
MODELS_TO_RUN = {
  'KNN': {'class': KNeighborsClassifier, 'params': {'n_neighbors': 5}},
  'GLQV': {'class': GlvqModel, 'params': {'prototypes_per_class': 1, 'max_iter': 2500, 'gtol': 1e-5, 'beta': 5, 'random_state': RANDOM_STATE}}
}

# Métricas
SCORES = {
  'accuracy': accuracy_score,
  'f1_score': f1_score,
  'g_mean': geometric_mean_score,
  'roc_auc': roc_auc_score
}

def calculate_score(y_true, y_pred) -> Dict[str, float]:
  """Calcula todas as métricas para um conjunto de predições."""
  results = [(name, func(y_true, y_pred)) for name, func in SCORES.items()]
  return dict(results)

def train_and_score_worker(task: Dict[str, Any], data_dir: str, folds: StratifiedKFold) -> Tuple[str, int, int, int, List[Dict[str, float]]]:
  """
  Função Worker que carrega o dataset e executa o K-Fold para uma única 
  combinação (Modelo, Dataset, Peso, Scaling).
  """
  # Desempacotar a tarefa
  model_name = task["model_name"]
  i = task["dataset_index"] # índice do dataset (0-99)
  j = task["weight_index"] # índice do peso (0-9)
  k = task["scaling_index"] # índice do scaling (0-2)
  weight_value = task["weight_value"]
  scaling_method = task["scaling_method"]
  
  # 1. Carregar o dataset específico
  try:
      # Caminho esperado: scaled_datasets/dataset_i+1/w_j.jjj/scaling_method.csv
      base_path = os.path.join(data_dir, f"dataset_{i+1}", f"w_{weight_value:.3f}")
      file_path = os.path.join(base_path, f"{scaling_method}.csv")
      dataset = pd.read_csv(file_path)
      X = dataset.iloc[:, :-1]
      y = dataset.iloc[:, -1]
  except Exception as e:
      print(f"Erro ao carregar o dataset (d={i+1}, w={weight_value:.3f}, s={scaling_method}): {e}", file=sys.stderr)
      # Retorna None para os scores para indicar falha
      return (model_name, i, j, k, None)
      
  # 2. Recriar/Clonar o objeto modelo
  # Instanciar um novo modelo é mais seguro em multiprocessing
  model_info = MODELS_TO_RUN[model_name]
  model_instance = model_info['class'](**model_info['params'])
  
  model_scores = []
  
  # 3. Executa a validação cruzada (K-Fold)
  # Usamos a mesma instância de StratifiedKFold em todos os workers (ela é 'picklable')
  for fold_index, (train_index, test_index) in enumerate(folds.split(X, y)):
      X_train = X.iloc[train_index]
      X_test = X.iloc[test_index]
      y_train = y.iloc[train_index]
      y_test = y.iloc[test_index]
      
      # Treinamento
      model_instance.fit(X_train, y_train)
      
      # Avaliação
      y_pred = model_instance.predict(X_test)
      model_scores.append(calculate_score(y_test, y_pred))
      
  print(f"Treino concluído: {model_name} | D={i+1} | W={weight_value:.3f} | S={scaling_method}", file=sys.stderr)
  
  # Retorna os identificadores e os resultados
  return (model_name, i, j, k, model_scores)


def run_parallel_training():
    # Calcula os pesos (necessário para os índices e nomes de arquivos)
    _, weights = generate_datasets.get_imbalance()
    num_datasets = 100 # Conforme documentado no generate_base_datasets.py
    
    # 1. Criação da lista de todas as tarefas a serem executadas
    all_tasks_config = []
    for i in range(num_datasets):
        for j, weight in enumerate(weights):
            for k, scaling_method in enumerate(SCALING_METHODS):
                for model_name in MODELS_TO_RUN.keys():
                    task = {
                        "model_name": model_name,
                        "dataset_index": i,
                        "weight_index": j,
                        "scaling_index": k,
                        "weight_value": weight,
                        "scaling_method": scaling_method,
                        "random_state": RANDOM_STATE
                    }
                    all_tasks_config.append(task)
                    
    # 2. Configuração do Pool de processos
    num_processes = cpu_count()
    total_tasks = len(all_tasks_config)
    print(f"Iniciando o treinamento paralelo de {total_tasks} combinações em {num_processes} processos.")
    print("Isso pode demorar dependendo da complexidade dos modelos (GLQV é lento).", file=sys.stderr)
    
    # Cria a instância de K-Fold para ser usada pelos workers
    folds_object = StratifiedKFold(n_splits=N_FOLDS, random_state=RANDOM_STATE, shuffle=True)
    
    # Converte a lista de dicionários para uma lista de tuplas de argumentos para starmap
    pool_args = [(task, DATASETS_DIR, folds_object) for task in all_tasks_config]
    
    # 3. Execução Paralela
    with Pool(num_processes) as pool:
        results_list = list(pool.starmap(train_and_score_worker, pool_args))

    print("Todos os treinamentos foram concluídos. Reestruturando resultados...")

    # 4. Reestruturação dos resultados
    all_results = {name: [
        [[None] * len(SCALING_METHODS) for _ in range(len(weights))] 
        for _ in range(num_datasets)
    ] for name in MODELS_TO_RUN.keys()}

    for result in results_list:
        model_name, i, j, k, scores = result
        if scores is not None:
            all_results[model_name][i][j][k] = scores
            
    # 5. Formatação e salvamento em CSV (Baseado na lógica original do notebook)
    models_frames = []
    for model_name in MODELS_TO_RUN.keys():
        datasets_frames = []
        for i in range(num_datasets):
            weights_frames = []
            for j in range(len(weights)):
                list_of_fold_results = all_results[model_name][i][j] # Lista de 3 elementos (scaling methods)
                
                scaling_frames = []
                for k, scores_for_scaling in enumerate(list_of_fold_results):
                    if scores_for_scaling is None:
                        # Se o treinamento falhou, adiciona um DataFrame vazio ou com NaNs
                        scaling_df = pd.DataFrame({m: [float('nan')]*N_FOLDS for m in SCORES.keys()}, index=[f"fold {f+1}" for f in range(N_FOLDS)])
                    else:
                        # scores_for_scaling é uma lista de 5 dicionários (folds)
                        scaling_df = pd.DataFrame(scores_for_scaling, index=[f"fold {f+1}" for f in range(N_FOLDS)])
                        
                    scaling_frames.append(scaling_df)

                # Concatena os resultados de scaling methods lado a lado
                weight_results = pd.concat(scaling_frames, axis=1, keys=SCALING_METHODS)
                weights_frames.append(weight_results)

            # Concatena os resultados de todos os pesos lado a lado
            dataset_results = pd.concat(weights_frames, axis=1, keys=[f"{weight:.3f}" for weight in weights])
            datasets_frames.append(dataset_results)
        
        # Concatena os resultados de todos os datasets lado a lado
        model_results = pd.concat(datasets_frames, axis=1, keys=[f"dataset {d+1}" for d in range(num_datasets)])
        models_frames.append(model_results)
    
    # Concatena os resultados de todos os modelos lado a lado
    final_result = pd.concat(models_frames, axis=1, keys=MODELS_TO_RUN.keys())
    
    # Salva o arquivo CSV
    final_result.to_csv("results.csv")
    print("\nResultados salvos em 'results.csv'.")
    
    # Mostra o cabeçalho do resultado final
    print("\nEstrutura do resultado final (primeiras 5 linhas e algumas colunas):")
    print(final_result.iloc[:, :10])
    
    return final_result


if __name__ == '__main__':
    # Esta verificação é crucial para o correto funcionamento do multiprocessing
    run_parallel_training()