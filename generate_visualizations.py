import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

# --- Configurações ---
RESULTS_DIR = "results_2"  # Pasta onde estão os arquivos .csv
OUTPUT_DIR = "plots"       # Pasta onde os gráficos serão salvos
METRIC_TO_PLOT = "accuracy" # Opções: accuracy, f1_score, g_mean, roc_auc

def generate_plots():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 1. Carregar e Consolidar todos os CSVs
    all_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    if not all_files:
        print(f"Nenhum arquivo CSV encontrado em {RESULTS_DIR}")
        return

    data_list = []
    print(f"Processando {len(all_files)} arquivos...")

    for file in all_files:
        try:
            # Carrega com os 5 níveis de colunas identificados no seu log
            df = pd.read_csv(file, header=[0, 1, 2, 3, 4], index_col=0)
            
            # Transforma em formato longo (Long Format)
            df_melted = df.melt(ignore_index=False).reset_index()
            df_melted.columns = ['Fold', 'Model', 'Dataset', 'Separation', 'Scaling', 'Metric', 'Score']
            
            # Converte separação para float para garantir ordem numérica no eixo X
            df_melted['Separation'] = df_melted['Separation'].astype(float)
            data_list.append(df_melted)
        except Exception as e:
            print(f"Erro ao processar {file}: {e}")

    full_df = pd.concat(data_list, ignore_index=True)
    
    # 2. Gráfico 1: Performance vs Separação (por Modelo)
    # Mostra como o classificador melhora conforme as classes se afastam
    for model in full_df['Model'].unique():
        plt.figure(figsize=(10, 6))
        model_data = full_df[(full_df['Model'] == model) & (full_df['Metric'] == METRIC_TO_PLOT)]
        
        sns.lineplot(data=model_data, x='Separation', y='Score', hue='Scaling', marker='o')
        
        plt.title(f"Desempenho ({METRIC_TO_PLOT}) vs Separação - {model}")
        plt.xlabel("Nível de Separação (Overlap)")
        plt.ylabel(METRIC_TO_PLOT.capitalize())
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Método de Escala', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"plot_sep_{model}.png"))
        plt.close()
        print(f"Gráfico de separação gerado para {model}.")

    # 3. Gráfico 2: Comparação Global de Scalers (Barra)
    # Média de todos os datasets e todas as separações
    plt.figure(figsize=(12, 6))
    global_data = full_df[full_df['Metric'] == METRIC_TO_PLOT]
    sns.barplot(data=global_data, x='Model', y='Score', hue='Scaling')
    plt.title(f"Comparação Global de Métodos de Escala por Modelo ({METRIC_TO_PLOT})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "global_comparison_bar.png"))
    plt.close()

    # 4. Gráfico 3: Heatmap (Modelo vs Scaling)
    # Médias gerais para uma visão rápida de qual scaler é melhor para qual modelo
    pivot_df = global_data.groupby(['Model', 'Scaling'])['Score'].mean().unstack()
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title(f"Heatmap de Performance Média ({METRIC_TO_PLOT})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "overall_heatmap.png"))
    plt.close()

    print(f"\nSucesso! Todos os gráficos foram salvos na pasta '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    generate_plots()