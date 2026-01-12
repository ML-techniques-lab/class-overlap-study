import pandas as pd
import random
import os
import sys
import numpy as np
import math
from multiprocessing import Pool, cpu_count
import time 

# Importação para clonagem de modelos (Necessário para segurança em multiprocessing)
from sklearn.base import clone 


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from deslib.des.knora_e import KNORAE
from deslib.dcs import OLA, LCA, MCB
from sklearn_lvq import GlvqModel
from xgboost import XGBClassifier
from deslib.des import KNORAU
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from imblearn.metrics import geometric_mean_score

from typing import Dict, Any, List, Tuple
import generate_datasets 
import generate_base_datasets # Necessário para pegar N_TESTS

# ==============================================================================
# SEÇÃO DE COMPATIBILIDADE (MONKEY PATCHES)
# ==============================================================================

# --- PATCH 1: CORREÇÃO SCIPY ---
import scipy.optimize
_original_minimize = scipy.optimize.minimize

def _patched_minimize(fun, x0, *args, **kwargs):
    x0 = np.asarray(x0)
    if x0.ndim > 1:
        x0 = x0.ravel()
    return _original_minimize(fun, x0, *args, **kwargs)

scipy.optimize.minimize = _patched_minimize

# --- PATCH 2: CORREÇÃO NUMPY ---
if not hasattr(np, 'math'):
    np.math = math

# ==============================================================================

# Importa o sklearn_lvq APÓS aplicar os patches
from sklearn_lvq import GlvqModel 

# --- Constantes e Configurações ---
SCALING_METHODS = ['original', 'SS', 'MA', 'RS', 'QT', 'PT']
DATASETS_DIR = "scaled_datasets"
# Usa o RANDOM_STATE do gerador base para consistência
RANDOM_STATE = generate_base_datasets.RANDOM_STATE 
N_FOLDS = 5

random.seed(RANDOM_STATE)

base_model = Perceptron(random_state=RANDOM_STATE)
# CORREÇÃO ANTERIOR APLICADA: 'base_estimator' -> 'estimator'
pool_classifiers = BaggingClassifier(estimator=base_model, n_estimators=100, random_state=RANDOM_STATE, bootstrap=True,
                                     bootstrap_features=False, max_features=1.0, n_jobs=-1)

# Modelos a serem rodados
MODELS_TO_RUN = {
  'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
  'SVM_lin': SVC(kernel='linear', probability=True),
  'SVM_rbf': SVC(kernel='rbf', probability=True),
  'GLQV': GlvqModel(prototypes_per_class=1, max_iter=2500, gtol=1e-5, beta=5, random_state=RANDOM_STATE),
  'LR': LogisticRegression(n_jobs=-1),
  'GNB': GaussianNB(),
  'GP': GaussianProcessClassifier(1.0 * RBF(1.0), random_state=RANDOM_STATE, n_jobs=-1),
  'LDA': LinearDiscriminantAnalysis(),
  'QDA': QuadraticDiscriminantAnalysis(),
  'DT': DecisionTreeClassifier(random_state=RANDOM_STATE),
  'MLP': MLPClassifier(activation='relu', solver='adam', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=RANDOM_STATE, max_iter=1000),
  'Percep': Perceptron(random_state=RANDOM_STATE, n_jobs=-1),
  'XGBoost': XGBClassifier(n_jobs=-1, random_state=RANDOM_STATE),
  'RF': RandomForestClassifier(random_state=0, n_jobs=-1),
  'AdaBoost': AdaBoostClassifier(n_estimators=100),
  'Bagging': pool_classifiers,
  'OLA': OLA(pool_classifiers, random_state=RANDOM_STATE),
  'LCA': LCA(pool_classifiers, random_state=RANDOM_STATE),
  'MCB': MCB(pool_classifiers, random_state=RANDOM_STATE),
  'KNORAE': KNORAE(pool_classifiers, random_state=RANDOM_STATE),
  'KNORAU': KNORAU(pool_classifiers, random_state=RANDOM_STATE)
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
  Função Worker que carrega o dataset e executa o K-Fold.
  """
  model_name = task["model_name"]
  i = task["dataset_index"] 
  j = task["sep_index"]      
  k = task["scaling_index"] 
  sep_value = task["sep_value"] 
  scaling_method = task["scaling_method"]
  
  try:
      # ADAPTAÇÃO DE CAMINHO:
      # Busca pasta dataset_X/sep_Y.YY/metodo.csv
      base_path = os.path.join(data_dir, f"dataset_{i+1}", f"sep_{sep_value:.2f}")
      file_path = os.path.join(base_path, f"{scaling_method}.csv")
      
      dataset = pd.read_csv(file_path)
      X = dataset.iloc[:, :-1]
      y = dataset.iloc[:, -1]
  except Exception as e:
      print(f"Erro ao carregar dataset (d={i+1}, sep={sep_value:.2f}): {e}", file=sys.stderr)
      return (model_name, i, j, k, None)
      
  # CORREÇÃO AQUI: Clonar o modelo diretamente.
  # Todos os modelos em MODELS_TO_RUN são instâncias (objetos prontos).
  # Usamos clone() para garantir que cada processo receba uma cópia isolada.
  try:
      model_instance = clone(MODELS_TO_RUN[model_name])
  except Exception as e:
      # Lida com falha de clonagem se o modelo for um tipo não-clonável (improvável para estes)
      print(f"Erro ao clonar modelo {model_name}: {e}", file=sys.stderr)
      return (model_name, i, j, k, None)

  model_scores = []
  
  # Executa a validação cruzada
  for fold_index, (train_index, test_index) in enumerate(folds.split(X, y)):
      X_train = np.ascontiguousarray(X.iloc[train_index].values, dtype=np.float64) 
      X_test = np.ascontiguousarray(X.iloc[test_index].values, dtype=np.float64)
      y_train = np.ascontiguousarray(y.iloc[train_index].values.ravel(), dtype=np.int32)
      y_test = np.ascontiguousarray(y.iloc[test_index].values.ravel(), dtype=np.int32)
      
      try:
          model_instance.fit(X_train, y_train)
          y_pred = model_instance.predict(X_test)
          model_scores.append(calculate_score(y_test, y_pred))
      except Exception as e:
          print(f"Erro no treino/predict ({model_name}, Fold {fold_index}): {e}", file=sys.stderr)
          model_scores.append(None) 

  print(f"Finalizado: {model_name} | Dataset {i+1} | Sep {sep_value:.2f}", file=sys.stderr)
  return (model_name, i, j, k, model_scores)


def run_parallel_training():
    # 1. Início do temporizador
    start_time = time.time()
    
    # ADAPTAÇÃO: Gera os valores de separação baseados nas constantes do generate_datasets
    try:
        sep_values = np.arange(generate_datasets.START_SEP, 
                               generate_datasets.END_SEP, 
                               generate_datasets.STEP_SEP)
    except Exception as e:
        print(f"Erro ao obter valores de separação: {e}", file=sys.stderr)
        return
        
    num_datasets = generate_base_datasets.N_TESTS
    
    all_tasks_config = []
    for i in range(num_datasets):
        # ADAPTAÇÃO: Loop sobre sep_values ao invés de weights
        for j, sep in enumerate(sep_values):
            for k, scaling_method in enumerate(SCALING_METHODS):
                for model_name in MODELS_TO_RUN.keys():
                    task = {
                        "model_name": model_name,
                        "dataset_index": i,
                        "sep_index": j,       # Alterado
                        "scaling_index": k,
                        "sep_value": sep,     # Alterado
                        "scaling_method": scaling_method,
                        "random_state": RANDOM_STATE
                    }
                    all_tasks_config.append(task)
                    
    #num_processes = max(1, int(cpu_count() * 0.75))
    
    num_processes = 6
    
    total_tasks = len(all_tasks_config)
    print(f"Iniciando processamento paralelo em {num_processes} núcleos.")
    print(f"Total de tarefas: {total_tasks}")
    
    folds_object = StratifiedKFold(n_splits=N_FOLDS, random_state=RANDOM_STATE, shuffle=True)
    pool_args = [(task, DATASETS_DIR, folds_object) for task in all_tasks_config]
    
    with Pool(num_processes) as pool:
        results_list = list(pool.starmap(train_and_score_worker, pool_args))

    print("Processamento concluído. Organizando resultados...")
    
    # 2. Fim do temporizador e cálculo do tempo
    end_time = time.time()
    total_time = end_time - start_time
    # --------------------------------------------------------

    # Estrutura para armazenar resultados (agora dimensionada por len(sep_values))
    all_results = {name: [
        [[None] * len(SCALING_METHODS) for _ in range(len(sep_values))] 
        for _ in range(num_datasets)
    ] for name in MODELS_TO_RUN.keys()}

    for result in results_list:
        model_name, i, j, k, scores = result
        if scores is not None:
            all_results[model_name][i][j][k] = scores
            
    # Formatação para DataFrame final
    models_frames = []
    for model_name in MODELS_TO_RUN.keys():
        datasets_frames = []
        for i in range(num_datasets):
            sep_frames = [] # Renomeado de weights_frames
            for j in range(len(sep_values)):
                list_of_fold_results = all_results[model_name][i][j]
                
                scaling_frames = []
                for k, scores_for_scaling in enumerate(list_of_fold_results):
                    if scores_for_scaling is None or any(s is None for s in scores_for_scaling) or len(scores_for_scaling) == 0:
                        scaling_df = pd.DataFrame({m: [np.nan]*N_FOLDS for m in SCORES.keys()}, 
                                                index=[f"fold {f+1}" for f in range(N_FOLDS)])
                    else:
                        scaling_df = pd.DataFrame(scores_for_scaling, 
                                                index=[f"fold {f+1}" for f in range(N_FOLDS)])
                    scaling_frames.append(scaling_df)

                # Concatena métodos de escala
                sep_results = pd.concat(scaling_frames, axis=1, keys=SCALING_METHODS)
                sep_frames.append(sep_results)

            # Concatena valores de separação (formatados com .2f)
            dataset_results = pd.concat(sep_frames, axis=1, keys=[f"{sep:.2f}" for sep in sep_values])
            datasets_frames.append(dataset_results)
        
        # Concatena datasets
        model_results = pd.concat(datasets_frames, axis=1, keys=[f"dataset {d+1}" for d in range(num_datasets)])
        models_frames.append(model_results)
    
    final_result = pd.concat(models_frames, axis=1, keys=MODELS_TO_RUN.keys())
    final_result.to_csv("results_overlap.csv")
    print("Salvo em 'results_overlap.csv'.")
    
    # 3. Impressão do tempo total
    print(f"\nTempo total de execução: {total_time:.2f} segundos ({total_time/60:.2f} minutos).")
    
    return final_result

if __name__ == '__main__':
    run_parallel_training()