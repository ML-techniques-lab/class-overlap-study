import subprocess
import time
import os
import sys

# --- Configurações ---

# Lista de todos os métodos de escala
SCALING_METHODS = ['original', 'SS', 'MA', 'RS', 'QT', 'PT']

# Lista de todos os modelos (Use os nomes exatos do seu MODELS dict)
MODELS = [
    'KNN', 'SVM_lin', 'SVM_rbf', 'GLQV', 'LR', 'GNB', 'GP', 'LDA', 'QDA', 'DT', 
    'MLP', 'Percep', 'XGBoost', 'RF', 'AdaBoost', 'Bagging', 'OLA', 'LCA', 
    'MCB', 'KNORAE', 'KNORAU'
]

# [MODIFICAÇÃO 1] Tempo de pausa em segundos (120 segundos = 2 minutos)
COOL_DOWN_TIME = 120 

# [MODIFICAÇÃO 2] Nome da pasta de resultados
RESULTS_DIR = "TASK_RESULTS"

# Nome do script que executa a tarefa real
TASK_SCRIPT = 'cluster_task_runner.py'

# --- Execução ---

def run_serial_tasks():
    # 1. [MODIFICAÇÃO 3] Cria a pasta de resultados se não existir
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Diretório de resultados criado: {RESULTS_DIR}")
        
    all_tasks = []
    # Gera a lista de todas as tarefas (Modelo x Escala)
    for model_name in MODELS:
        for scale_method in SCALING_METHODS:
            all_tasks.append((model_name, scale_method))
            
    total_tasks = len(all_tasks)
    print(f"Total de {total_tasks} tarefas geradas. Iniciando execução serial...")
    
    # Loop principal de execução
    for index, (model_name, scale_method) in enumerate(all_tasks):
        
        # [MODIFICAÇÃO 4] Define o caminho completo do arquivo de saída dentro da pasta
        file_name = f"results_task_{index+1}_{model_name}_{scale_method}.csv"
        outfile = os.path.join(RESULTS_DIR, file_name)
        
        # Argumentos para o cluster_task_runner.py: <scale> <model> <outfile>
        command = [
            'python', TASK_SCRIPT,
            scale_method,
            model_name,
            outfile
        ]
        
        task_info = f"[{index+1}/{total_tasks}] Modelo: {model_name}, Escala: {scale_method}"
        print("-" * 50)
        print(f"Executando Tarefa {task_info}")
        print(f"Comando: {' '.join(command)}")

        # Executa o job
        try:
            start_time = time.time()
            # O 'cluster_task_runner.py' irá salvar o CSV no caminho especificado por 'outfile'
            subprocess.run(command, check=True) 
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Tarefa {task_info} concluída em {elapsed_time:.2f} segundos. Salvo em: {outfile}")
            
        except subprocess.CalledProcessError as e:
            print(f"ERRO: A tarefa {task_info} falhou. Verifique logs: {outfile}", file=sys.stderr)
            continue 
        except FileNotFoundError:
             print(f"ERRO: O script {TASK_SCRIPT} não foi encontrado. Verifique o caminho.", file=sys.stderr)
             sys.exit(1)


        # Pausa para Resfriamento
        if index < total_tasks - 1:
            print(f"Aguardando {COOL_DOWN_TIME} segundos para resfriamento da CPU...")
            time.sleep(COOL_DOWN_TIME)
            
    print("\nTODAS AS TAREFAS SERIAIS CONCLUÍDAS.")

if __name__ == '__main__':
    run_serial_tasks()