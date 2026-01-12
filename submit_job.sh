#!/bin/bash
#SBATCH --job-name=OverlapExp
#SBATCH --account=def-menelau      # SUBSTITUA PELO SEU ACCOUNT
#SBATCH --array=1-10                   # Se quer 10 jobs de 10 datasets cada
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G                       # Aumente se o SVM der erro de memória
#SBATCH --time=08:00:00                # Tempo suficiente para os modelos lentos
#SBATCH --output=logs/exp_%A_%a.out    # Salva logs de erro/saída

# Carrega ambiente
source venv/bin/activate

# Lógica de fatiamento (slice)
step=$((100 / $SLURM_ARRAY_TASK_MAX))
start=$(( ((($SLURM_ARRAY_TASK_ID - 1) * step) + 1) ))
end=$(( $SLURM_ARRAY_TASK_ID * step ))

# Execução (usando o nome correto do arquivo e passando os argumentos)
# Exemplo: sbatch script.sh all KNN
python cluster_task_runner.py all ${1} $start:$end "results_2/${1}_${SLURM_ARRAY_TASK_ID}.csv"