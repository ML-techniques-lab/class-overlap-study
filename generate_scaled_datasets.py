#!/usr/bin/env python
# -*- coding: utf-8 -*-

import generate_datasets
import generate_base_datasets
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DATASETS_DIR = "datasets"
OUT_DIR = "scaled_datasets"
# Recupera o RANDOM_STATE base
RANDOM_STATE = generate_base_datasets.RANDOM_STATE

# Recupera as configurações de separação do arquivo que gerou os dados
START_SEP = generate_datasets.START_SEP
END_SEP = generate_datasets.END_SEP
STEP_SEP = generate_datasets.STEP_SEP
N_TESTS = generate_base_datasets.N_TESTS

if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

datasets = []

# Gera os mesmos valores de separação usados na geração
sep_values = np.arange(START_SEP, END_SEP, STEP_SEP)

print(f"Iniciando leitura de {N_TESTS} datasets com {len(sep_values)} variações de overlap cada...")

# Itera numericamente para garantir que dataset_1 seja processado como dataset_1
for i in range(N_TESTS):
    dataset_folder_name = f"dataset_{i+1}"
    dataset_path = os.path.join(DATASETS_DIR, dataset_folder_name)
    
    # Monta os nomes dos arquivos baseados no padrão de separação (ex: dataset_1_sep_0.10.csv)
    # Nota: O formato .2f deve coincidir com o usado em generate_datasets.py
    dataset_files = [f"{dataset_folder_name}_sep_{sep:.2f}.csv" for sep in sep_values]
    
    current_dataset_variations = []
    for file_name in dataset_files:
        full_path = os.path.join(dataset_path, file_name)
        try:
            df = pd.read_csv(full_path)
            current_dataset_variations.append(df)
        except FileNotFoundError:
            print(f"Aviso: Arquivo não encontrado: {full_path}")
            # Adiciona None ou lida com erro se necessário, 
            # mas assume-se que os arquivos foram gerados corretamente.
    
    if current_dataset_variations:
        datasets.append(current_dataset_variations)

scalers = {
    'MM': MinMaxScaler(),
    'SS': StandardScaler()
}

results = {}

print("Aplicando escalonamento...")

for name, scaler in scalers.items():
    scaled_datasets = []
    # Para cada conjunto de dados (1 a 100)
    for i in range(len(datasets)):
        scaled_variations = []
        # Para cada variação de overlap dentro do dataset
        for j in range(len(datasets[i])):
            dataset = datasets[i][j]
            
            # Separa X e y
            X = dataset.iloc[:, :-1]
            y = dataset.iloc[:, -1]
            
            # Aplica o scaler apenas no X
            X_scaled = scaler.fit_transform(X)
            
            # Reconstrói o DataFrame
            scaled_dataset = pd.concat([pd.DataFrame(X_scaled, columns=dataset.columns[:-1]), y], axis=1)
            scaled_variations.append(scaled_dataset)
        scaled_datasets.append(scaled_variations)
    results[name] = scaled_datasets

print("Salvando arquivos...")

for i in range(len(datasets)):
    # Cria a estrutura de pasta para o dataset atual (ex: scaled_datasets/dataset_1)
    dataset_out_path = f"{OUT_DIR}/dataset_{i+1}"
    if not os.path.exists(dataset_out_path):
        os.makedirs(dataset_out_path)

    for j in range(len(sep_values)):
        # Cria a subpasta para o nível de separação (ex: sep_0.10)
        sep_val = sep_values[j]
        base_path = f"{dataset_out_path}/sep_{sep_val:.2f}"
        
        if not os.path.exists(base_path):
            os.makedirs(base_path)
            
        # Salva o original (sem escala) nesta nova estrutura organizada
        datasets[i][j].to_csv(f"{base_path}/original.csv", index=False)
        
        # Salva as versões escalonadas
        for name, result in results.items():
            results[name][i][j].to_csv(f"{base_path}/{name}.csv", index=False)

print("Concluído.")