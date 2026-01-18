"""Imports"""
# Core
import pandas as pd
import random
import sys
import os
import numpy as np
from sklearn.base import clone
import time
from sklearn.utils.validation import check_X_y

# Dependências do projeto
import generate_datasets 
import generate_base_datasets

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
import sklearn.base
from deslib.base import BaseDS as DES
from sklearn.svm import SVC, LinearSVC

from sklearn.calibration import CalibratedClassifierCV 


#====================
def sklearn_validate_patch(self, X, y, **kwargs):
    # Simula o comportamento do _validate_data que o DESlib espera
    return check_X_y(X, y, accept_sparse=True)
#======================

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

base_model = Perceptron(random_state=RANDOM_STATE, max_iter=1000)
pool_classifiers = BaggingClassifier(estimator=base_model, n_estimators=100, random_state=RANDOM_STATE)

from sklearn.datasets import make_classification
X_dummy, y_dummy = make_classification(n_samples=10, n_features=4)
pool_classifiers.fit(X_dummy, y_dummy)

MODELS = {
  'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=1),
  'SVM_lin': CalibratedClassifierCV(LinearSVC(dual='auto', random_state=RANDOM_STATE)), 
  'SVM_rbf': SVC(kernel='rbf', probability=True),
  'GLQV': GlvqModel(prototypes_per_class=1, max_iter=2500, gtol=1e-5, beta=5, random_state=RANDOM_STATE, display=False),  
  'LR': LogisticRegression(n_jobs=1),
  'GNB': GaussianNB(),
  'GP': GaussianProcessClassifier(1.0 * RBF(1.0), random_state=RANDOM_STATE, n_jobs=1),
  'LDA': LinearDiscriminantAnalysis(),
  'QDA': QuadraticDiscriminantAnalysis(),
  'DT': DecisionTreeClassifier(random_state=RANDOM_STATE),
  'MLP': MLPClassifier(activation='relu', solver='adam', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=RANDOM_STATE, max_iter=1000),
  'Percep': Perceptron(random_state=RANDOM_STATE, n_jobs=1),
  'XGBoost': XGBClassifier(n_jobs=1, random_state=RANDOM_STATE),
  'RF': RandomForestClassifier(random_state=0, n_jobs=1),
  'AdaBoost': AdaBoostClassifier(n_estimators=100),
  'Bagging': pool_classifiers,
  'OLA': OLA(pool_classifiers, random_state=RANDOM_STATE),
  'LCA': LCA(pool_classifiers, random_state=RANDOM_STATE),
  'MCB': MCB(pool_classifiers, random_state=RANDOM_STATE),
  'KNORAE': KNORAE(pool_classifiers, random_state=RANDOM_STATE),
  'KNORAU': KNORAU(pool_classifiers, random_state=RANDOM_STATE)
}

# fazer o fit primeiro do bagging, antes

random.seed(RANDOM_STATE)

"""System arguments"""
if len(sys.argv) < 4:
    print("Uso: python cluster_task_runner.py <scaling> <models> <start:end> [output_file]")
    sys.exit(1)

args = sys.argv
# Escalas
scaling_methods_to_run = SCALING_METHODS if args[1].lower() == 'all' else args[1].split(',')
# Modelos
models_to_run = list(MODELS.keys()) if args[2].lower() == 'all' else args[2].split(',')

# --- LÓGICA DE SLICE DE DATASETS ---
try:
    ds_range = args[3].split(':')
    ds_start = int(ds_range[0]) - 1  # Converte para índice 0 (ex: "1" vira 0)
    ds_end = int(ds_range[1])       # O range do Python já é exclusivo no final
except:
    print("Erro: O formato do intervalo deve ser start:end (ex: 1:10)")
    sys.exit(1)

if len(args) == 5:
    outfile = args[4]
else:
    outfile = f"results_{args[2]}_{args[3].replace(':','-')}.csv"

"""Load datasets"""
sep_values = np.arange(generate_datasets.START_SEP, generate_datasets.END_SEP, generate_datasets.STEP_SEP)
datasets = []

# Carregamos apenas os datasets do intervalo solicitado
print(f"Carregando datasets do índice {ds_start+1} ao {ds_end}...")
for i in range(ds_start, ds_end):
    base_path = f"{DATASETS_DIR}/dataset_{i+1}"
    levels = []
    for j, sep in enumerate(sep_values):
        path = os.path.join(base_path, f"sep_{sep:.2f}") 
        methods = []
        
        if not os.path.exists(path):
            methods = [None] * len(scaling_methods_to_run)
        else:
            for scaling_method in scaling_methods_to_run:
                file_path = os.path.join(path, f"{scaling_method}.csv")
                try:
                    methods.append(pd.read_csv(file_path))
                except:
                    methods.append(None)
        levels.append(methods)
    datasets.append(levels)

def calculate_score(y_true, y_pred):
    results = [(name, func(y_true, y_pred)) for name, func in SCORES.items()]
    return dict(results)

"""Train models"""
results = {name: [] for name in models_to_run}
folds = StratifiedKFold(n_splits=N_FOLDS, random_state=RANDOM_STATE, shuffle=True)
start_time = time.time()

for name in models_to_run:
    model = MODELS[name]
    print(f"Treinando: {name}")
    
    for i in range(len(datasets)): # Este loop agora só percorre os datasets carregados (o slice)
        dataset_results = []
        for j in range(len(sep_values)):
            level_results = []
            for k in range(len(scaling_methods_to_run)):
                dataset = datasets[i][j][k]
                if dataset is None:
                    level_results.append([None] * N_FOLDS)
                    continue
                
                X = dataset.iloc[:, :-1]
                y = dataset.iloc[:, -1]
                model_scores = []
                current_model_instance = clone(model)
                
                for train_index, test_index in folds.split(X, y):
                    if (name == 'GLQV'):
                        # GLQV exige float64 e arrays NumPy contíguos (flattened)
                        X_train = X.iloc[train_index].values.astype(np.float64)
                        X_test = X.iloc[test_index].values.astype(np.float64)
                        y_train = y.iloc[train_index].values.astype(np.int64)
                        y_test = y.iloc[test_index].values.astype(np.int64)
                    else:
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    if (name in ['OLA', 'LCA', 'MCB', 'KNORAE', 'KNORAU']):
                        # Conversão obrigatória para Numpy (evita erros de índice e FutureWarnings no Python 3.14)
                        X_train_np = X_train.to_numpy()
                        y_train_np = y_train.to_numpy().ravel()

                        current_model_instance.pool_classifiers.fit(X_train_np, y_train_np)

                        current_model_instance._validate_data = sklearn_validate_patch.__get__(current_model_instance)

                        # Agora o fit funcionará e criará o atributo 'estimators_'
                        current_model_instance.fit(X_train_np, y_train_np)

                    
                import traceback

                try:

                    current_model_instance.fit(X_train, y_train)
                    y_pred = current_model_instance.predict(X_test)
                    model_scores.append(calculate_score(y_test, y_pred))
                except Exception as e:
                    print(f"ERRO no modelo {name}: {e}")
                    traceback.print_exc()  # This will show exactly where 'estimators_' was called
                    model_scores.append(None)
                
                level_results.append(model_scores)
            dataset_results.append(level_results)
        results[name].append(dataset_results)

"""Saving results"""
total_time = time.time() - start_time
models_frames = []
actual_dataset_indices = range(ds_start + 1, ds_end + 1)

for model_name in models_to_run:
    datasets_frames = []
    for i in range(len(datasets)):
        sep_frames = []
        for j in range(len(sep_values)):
            list_of_fold_results = results[model_name][i][j]
            score_frames = []
            for scores in list_of_fold_results:
                if scores is None or any(s is None for s in scores):
                    df = pd.DataFrame({m: [np.nan]*N_FOLDS for m in SCORES.keys()}, 
                                      index=[f"fold {f+1}" for f in range(N_FOLDS)])
                else:
                    df = pd.DataFrame(scores, index=[f"fold {f+1}" for f in range(N_FOLDS)])
                score_frames.append(df)

            sep_frames.append(pd.concat(score_frames, axis=1, keys=scaling_methods_to_run))
        
        datasets_frames.append(pd.concat(sep_frames, axis=1, keys=[f"{sep:.2f}" for sep in sep_values]))
    
    # Aqui ajustamos as chaves para refletir os números reais dos datasets (ex: dataset 11, 12...)
    model_results = pd.concat(datasets_frames, axis=1, keys=[f"dataset {d}" for d in actual_dataset_indices])
    models_frames.append(model_results)

final_result = pd.concat(models_frames, axis=1, keys=models_to_run)
final_result.to_csv(outfile)
print(f"\nFinalizado! Salvo em: {outfile} em {total_time/60:.2f} min.")