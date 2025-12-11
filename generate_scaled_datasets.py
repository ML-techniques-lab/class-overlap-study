#!/usr/bin/env python
# -*- coding: utf-8 -*-

import generate_datasets
import generate_base_datasets
import pandas as pd
import os
import numpy as np
import time # Adicionado para o temporizador
import sys # Adicionado para impressões de erro/status no worker
from multiprocessing import Pool, cpu_count # Adicionado para paralelismo

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, \
  RobustScaler, QuantileTransformer, PowerTransformer

# --- Constantes originais ---
DATASETS_DIR = "datasets"
OUT_DIR = "scaled_datasets"
RANDOM_STATE = generate_base_datasets.RANDOM_STATE
START_SEP = generate_datasets.START_SEP
END_SEP = generate_datasets.END_SEP
STEP_SEP = generate_datasets.STEP_SEP
N_TESTS = generate_base_datasets.N_TESTS

# --- Definindo Scalers (fora da função principal) ---
SCALERS = {
    'MM': MinMaxScaler(),
    'SS': StandardScaler(),
    'MA': MaxAbsScaler(),
    'RS': RobustScaler(),
    'QT': QuantileTransformer(random_state=RANDOM_STATE),
    'PT': PowerTransformer(method='yeo-johnson', standardize=True)
}

# ------------------------------------------------------------------------------
# FUNÇÃO WORKER PARA PROCESSAMENTO PARALELO
# ------------------------------------------------------------------------------
def scale_and_save_worker(i: int, sep: float, scalers: dict, datasets_dir: str, out_dir: str):
    """
    Carrega um dataset específico (dataset_i, sep_j), aplica todos os scalers 
    e salva os resultados na estrutura de pastas de saída.
    """
    # 1. Definir caminhos de entrada e saída
    dataset_folder_name = f"dataset_{i+1}"
    file_name = f"{dataset_folder_name}_sep_{sep:.2f}.csv"
    full_path = os.path.join(datasets_dir, dataset_folder_name, file_name)
    
    # Caminho base de saída (ex: scaled_datasets/dataset_1/sep_0.10)
    base_out_path = os.path.join(out_dir, dataset_folder_name, f"sep_{sep:.2f}")

    # 2. Carregar Dados
    try:
        df = pd.read_csv(full_path)
    except FileNotFoundError:
        print(f"Worker: Arquivo não encontrado: {full_path}", file=sys.stderr)
        return False

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 3. Criar estrutura de diretório de saída
    if not os.path.exists(base_out_path):
        os.makedirs(base_out_path)

    # 4. Salvar o original (sem escala)
    df.to_csv(os.path.join(base_out_path, "original.csv"), index=False)

    # 5. Aplicar e salvar as versões escalonadas
    for name, scaler in scalers.items():
        try:
            # Clona o scaler para garantir que cada worker tenha sua própria instância
            # e fit/transform seja isolado
            scaler_instance = scaler.__class__()
            # A classe QuantileTransformer precisa do random_state para ser reproduzível
            if name == 'QT':
                 scaler_instance.set_params(random_state=RANDOM_STATE) 
            
            X_scaled = scaler_instance.fit_transform(X)
            
            scaled_dataset = pd.concat([pd.DataFrame(X_scaled, columns=df.columns[:-1]), y], axis=1)
            scaled_dataset.to_csv(os.path.join(base_out_path, f"{name}.csv"), index=False)
        except Exception as e:
            # Imprime o erro no stderr para não interferir com a saída padrão
            print(f"Worker: Falha ao aplicar {name} em {file_name}: {e}", file=sys.stderr)
            
    print(f"Worker Concluído: Dataset {i+1}, Sep {sep:.2f}", file=sys.stderr)
    return True # Retorna sucesso

# ------------------------------------------------------------------------------
# FUNÇÃO PRINCIPAL DE EXECUÇÃO PARALELA
# ------------------------------------------------------------------------------
def run_scaling_parallel():
    
    # --- TEMPORIZADOR: INÍCIO ---
    start_time = time.time() 
    
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    sep_values = np.arange(START_SEP, END_SEP, STEP_SEP)

    # 1. Criação da lista de tarefas (Task = (dataset_i, sep_j))
    all_tasks = []
    for i in range(N_TESTS):
        for sep in sep_values:
            # Tupla de argumentos para o worker
            all_tasks.append((i, sep, SCALERS, DATASETS_DIR, OUT_DIR))

    # 2. Configuração do Pool de Processos
    num_processes = cpu_count()
    total_tasks = len(all_tasks)
    
    print(f"Iniciando escalonamento paralelo de {total_tasks} arquivos em {num_processes} núcleos.")
    
    # 3. Execução Paralela
    try:
        with Pool(num_processes) as pool:
            # starmap distribui a lista de tuplas de argumentos para a função worker
            pool.starmap(scale_and_save_worker, all_tasks)
    except Exception as e:
        print(f"Erro fatal na execução paralela: {e}", file=sys.stderr)
        
    # --- TEMPORIZADOR: FIM ---
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\nProcessamento concluído.")
    print(f"Tempo total de execução: {total_time:.2f} segundos ({total_time/60:.2f} minutos).")


if __name__ == '__main__':
    # Esta verificação é crucial para o correto funcionamento do multiprocessing
    run_scaling_parallel()