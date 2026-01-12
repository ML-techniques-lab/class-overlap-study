"""Imports"""
# Core
import pandas as pd
import random
import sys
import os
import numpy as np # Adicionado
from sklearn.base import clone # Adicionado para clonar modelos (essencial)
import time

# Dependências do projeto (Acesso às constantes de overlap)
import generate_datasets 
import generate_base_datasets # Adicionado

# Metrics and model selection
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from imblearn.metrics import geometric_mean_score

# Models
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

"""Constants"""

DATASETS_DIR = "scaled_datasets"
RANDOM_STATE = 10
N_FOLDS = 5
SCALING_METHODS = ['original', 'SS', 'MA', 'RS', 'QT', 'PT']
SCORES = {
  'accuracy': accuracy_score,
  'f1_score': f1_score,
  'g_mean': geometric_mean_score,
  'roc_auc': roc_auc_score
}

base_model = Perceptron(random_state=RANDOM_STATE, max_iter=1000) # Ajustado max_iter
# CORREÇÃO: base_estimator substituído por estimator (para scikit-learn >= 1.2)
pool_classifiers = BaggingClassifier(estimator=base_model, n_estimators=100, random_state=RANDOM_STATE, bootstrap=True,
                                     bootstrap_features=False, max_features=1.0, n_jobs=1) # n_jobs=1 (serial no worker)
MODELS = {
  'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=1), # n_jobs=1
  'SVM_lin': SVC(kernel='linear', probability=True),
  'SVM_rbf': SVC(kernel='rbf', probability=True),
  'GLQV': GlvqModel(prototypes_per_class=1, max_iter=2500, gtol=1e-5, beta=5, random_state=RANDOM_STATE),
  'LR': LogisticRegression(n_jobs=1), # n_jobs=1
  'GNB': GaussianNB(),
  'GP': GaussianProcessClassifier(1.0 * RBF(1.0), random_state=RANDOM_STATE, n_jobs=1), # n_jobs=1
  'LDA': LinearDiscriminantAnalysis(),
  'QDA': QuadraticDiscriminantAnalysis(),
  'DT': DecisionTreeClassifier(random_state=RANDOM_STATE),
  'MLP': MLPClassifier(activation='relu', solver='adam', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=RANDOM_STATE, max_iter=1000), # max_iter ajustado
  'Percep': Perceptron(random_state=RANDOM_STATE, n_jobs=1), # n_jobs=1
  'XGBoost': XGBClassifier(n_jobs=1, random_state=RANDOM_STATE), # n_jobs=1
  'RF': RandomForestClassifier(random_state=0, n_jobs=1), # n_jobs=1
  'AdaBoost': AdaBoostClassifier(n_estimators=100),
  'Bagging': pool_classifiers,
  'OLA': OLA(pool_classifiers, random_state=RANDOM_STATE),
  'LCA': LCA(pool_classifiers, random_state=RANDOM_STATE),
  'MCB': MCB(pool_classifiers, random_state=RANDOM_STATE),
  'KNORAE': KNORAE(pool_classifiers, random_state=RANDOM_STATE),
  'KNORAU': KNORAU(pool_classifiers, random_state=RANDOM_STATE)
}
# Nota: n_jobs ajustado para 1 em modelos que o suportam, pois a paralelização externa é feita pelo cluster.

random.seed(RANDOM_STATE)

"""System arguments"""
# Verifica se os argumentos de linha de comando foram fornecidos
if len(sys.argv) < 3:
    print("Uso: python cluster_task_runner.py <scaling_methods|all> <models|all> [output_file]")
    sys.exit(1)

args = sys.argv
# Processa métodos de escala
if (args[1].lower() == 'all'):
  scaling_methods_to_run = SCALING_METHODS
else:
  scaling_methods_to_run = args[1].split(',')
# Processa modelos
if (args[2].lower() == 'all'):
  models_to_run = MODELS.keys()
else:
  models_to_run = args[2].split(',')

if len(args) == 4:
  outfile = args[3]
else:
  # Garante que o nome de saída reflita o subconjunto, se não for especificado
  scaling_str = "all_scales" if args[1].lower() == 'all' else "_".join(scaling_methods_to_run)
  models_str = "all_models" if args[2].lower() == 'all' else "_".join(models_to_run)
  outfile = f'results_{scaling_str}_{models_str}.csv'

"""Load datasets (Adaptado para a estrutura de Overlap)"""
sep_values = np.arange(generate_datasets.START_SEP, generate_datasets.END_SEP, generate_datasets.STEP_SEP)
num_datasets = generate_base_datasets.N_TESTS 

datasets = []
print(f"Carregando {num_datasets} datasets...")

# Itera sobre datasets (0 a 99)
for i in range(num_datasets):
    base_path = f"{DATASETS_DIR}/dataset_{i+1}"
    levels = []
    # Itera sobre os níveis de separação (overlap)
    for j, sep in enumerate(sep_values):
        # CORREÇÃO: Path agora usa 'sep_0.xx' e não 'w_0.xxx'
        path = os.path.join(base_path, f"sep_{sep:.2f}") 
        
        methods = []
        if not os.path.exists(path):
            print(f"Aviso: Diretório não encontrado: {path}", file=sys.stderr)
            methods = [None] * len(scaling_methods_to_run)
            levels.append(methods)
            continue
            
        # Itera sobre os métodos de escala selecionados
        for scaling_method in scaling_methods_to_run:
            file_name = scaling_method + ".csv"
            try:
                # Carrega o arquivo específico
                methods.append(pd.read_csv(os.path.join(path, file_name)))
            except FileNotFoundError:
                print(f"Erro: Arquivo não encontrado: {os.path.join(path, file_name)}", file=sys.stderr)
                methods.append(None) # Adiciona None em caso de falha de carregamento
        levels.append(methods)
    datasets.append(levels)

def calculate_score(y_true, y_pred):
    results = [(name, func(y_true, y_pred)) for name, func in SCORES.items()]
    return dict(results)

"""Train models"""
results = {name: [] for name in models_to_run}
folds = StratifiedKFold(n_splits=N_FOLDS, random_state=RANDOM_STATE, shuffle=True)
start_time = time.time() # Adiciona temporizador

# Itera sobre os modelos SELECIONADOS
for name in models_to_run:
    model = MODELS[name]
    print(f"Iniciando treinamento do modelo: {name}")
    
    for i in range(len(datasets)): # Loop Dataset
        dataset_results = []
        for j in range(len(sep_values)): # Loop Sep Value
            level_results = []
            
            # NOTA: O 'k' aqui mapeia para o índice dentro de 'scaling_methods_to_run'
            for k in range(len(scaling_methods_to_run)): # Loop Scaling Method
                dataset = datasets[i][j][k]
                
                if dataset is None:
                    level_results.append([None] * N_FOLDS) # Adiciona resultados nulos se o dataset falhou
                    continue
                    
                X = dataset.iloc[:, :-1]
                y = dataset.iloc[:, -1]
                model_scores = []
                
                # CORREÇÃO: Clona o modelo ANTES do K-Fold (melhor: antes do scaling loop)
                # Garante que cada combinação (Dataset, Sep, Scaling) use um modelo novo.
                current_model_instance = clone(model)
                
                for train_index, test_index in folds.split(X, y):
                    X_train = X.iloc[train_index]
                    X_test = X.iloc[test_index]
                    y_train = y.iloc[train_index]
                    y_test = y.iloc[test_index]
                    
                    try:
                        current_model_instance.fit(X_train, y_train)
                        y_pred = current_model_instance.predict(X_test)
                        model_scores.append(calculate_score(y_test, y_pred))
                    except Exception as e:
                        print(f"Erro no treino/predict ({name}, D{i+1}, Sep{sep_values[j]:.2f}, Fold): {e}", file=sys.stderr)
                        model_scores.append(None) # Adiciona None em caso de falha
                        
                level_results.append(model_scores)
            dataset_results.append(level_results)
        results[name].append(dataset_results)
    print(f"Modelo {name} concluído.")

"""Saving results"""
end_time = time.time() # Fim do temporizador
total_time = end_time - start_time

models_frames = []
for model_name in models_to_run:
    datasets_frames = []
    for i in range(len(datasets)):
        sep_frames = []
        for j, sep in enumerate(sep_values):
            # Obtém a lista de scores para todos os métodos de escala selecionados
            list_of_fold_results = results[model_name][i][j]
            
            score_frames = []
            for scores in list_of_fold_results:
                 # Cria o DataFrame para cada método de escala (6 colunas = 5 folds x 4 métricas)
                 if scores is None or any(s is None for s in scores):
                     # Adiciona DataFrame de NaNs se o treinamento falhou
                     scaling_df = pd.DataFrame({m: [np.nan]*N_FOLDS for m in SCORES.keys()}, 
                                             index=[f"fold {f+1}" for f in range(N_FOLDS)])
                 else:
                     scaling_df = pd.DataFrame(scores, index=[f"fold {f+1}" for f in range(N_FOLDS)])
                 
                 score_frames.append(scaling_df)

            # Concatena métodos de escala
            weight_results = pd.concat(score_frames, axis=1, keys=scaling_methods_to_run)
            sep_frames.append(weight_results)
            
        # Concatena valores de separação
        dataset_results = pd.concat(sep_frames, axis=1, keys=[f"{sep:.2f}" for sep in sep_values])
        datasets_frames.append(dataset_results)
        
    # Concatena datasets
    model_results = pd.concat(datasets_frames, axis=1, keys=[f"dataset {d+1}" for d in range(num_datasets)])
    models_frames.append(model_results)

final_result = pd.concat(models_frames, axis=1, keys=models_to_run)
final_result.to_csv(outfile)
print(f"\nResultados salvos em '{outfile}'.")
print(f"Tempo total de execução do(s) subconjunto(s) da tarefa: {total_time:.2f} segundos ({total_time/60:.2f} minutos).")