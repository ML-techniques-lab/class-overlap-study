import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

# --- Configurações ---
RESULTS_DIR = "results_2"  # Pasta com os CSVs
OUTPUT_DIR = "plots_granular"
SCALERS_ORDER = ['original', 'SS', 'MA', 'RS', 'QT', 'PT']
METRICS = ['accuracy', 'f1_score', 'g_mean', 'roc_auc']

def load_and_consolidate_data():
    """Lê todos os CSVs e converte para um formato longo (Tidy Data)."""
    all_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    if not all_files:
        print(f"Nenhum arquivo encontrado em {RESULTS_DIR}")
        return None

    print(f"Lendo {len(all_files)} arquivos CSV...")
    data_list = []
    
    for file in all_files:
        try:
            # Lê o cabeçalho multinível (Model, Dataset, Sep, Scaler, Metric)
            df = pd.read_csv(file, header=[0, 1, 2, 3, 4], index_col=0)
            
            # Melt para transformar colunas em linhas
            df_melted = df.melt(ignore_index=False).reset_index()
            df_melted.columns = ['Fold', 'Model', 'Dataset', 'Separation', 'Scaling', 'Metric', 'Score']
            
            # Conversões de tipo
            df_melted['Separation'] = pd.to_numeric(df_melted['Separation'], errors='coerce')
            df_melted['Score'] = pd.to_numeric(df_melted['Score'], errors='coerce')
            
            # Extrai o número do dataset para ordenação (ex: "dataset 10" -> 10)
            df_melted['Dataset_Num'] = df_melted['Dataset'].str.extract('(\d+)').astype(float)
            
            data_list.append(df_melted)
        except Exception as e:
            print(f"Erro ao ler {file}: {e}")

    if not data_list:
        return None
        
    print("Consolidando dados...")
    return pd.concat(data_list, ignore_index=True)

def plot_global_averages(df):
    """
    Gera gráficos de LINHAS comparando a performance vs separação.
    Nesta versão, a média é calculada sobre todos os datasets para cada modelo.
    """
    print("\n--- Gerando Gráficos de Linha (Performance vs Separação) ---")
    
    # Criar subpasta específica para as médias por modelo
    save_path_root = os.path.join(OUTPUT_DIR, "Global_Averages")
    
    for metric in METRICS:
        metric_df = df[df['Metric'] == metric]
        if metric_df.empty:
            continue
            
        # Pasta por métrica
        metric_path = os.path.join(save_path_root, metric)
        os.makedirs(metric_path, exist_ok=True)
        
        # Para cada modelo, geramos um gráfico de linha consolidando todos os seus datasets
        unique_models = metric_df['Model'].unique()
        
        for model in unique_models:
            model_data = metric_df[metric_df['Model'] == model]
            
            plt.figure(figsize=(10, 6))
            
            # O lineplot com 'hue' no Scaling calculará automaticamente a média e o 
            # intervalo de confiança considerando todos os datasets e folds do modelo.
            sns.lineplot(
                data=model_data, 
                x='Separation', 
                y='Score', 
                hue='Scaling', 
                hue_order=SCALERS_ORDER,
                marker='o', 
                linewidth=2,
                palette="viridis"
            )
            
            plt.title(f"Média Geral: {model} - {metric.upper()} vs Separação")
            plt.xlabel("Nível de Separação (Quanto maior, menor o overlap)")
            plt.ylabel(f"Score Médio ({metric.capitalize()})")
            plt.ylim(0, 1.05)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(title='Método de Escala', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            filename = f"avg_lineplot_{model}_{metric}.pdf"
            plt.savefig(os.path.join(metric_path, filename))
            plt.close()
            
        print(f"Gráficos de linha (Performance vs Sep) salvos para a métrica: {metric}")

    # (Opcional) Manter a geração de Heatmaps em uma pasta separada, 
    # pois eles são ótimos para o resumo final do artigo.
    print("Gerando Heatmaps resumidos...")
    for metric in METRICS:
        metric_df = df[df['Metric'] == metric]
        if metric_df.empty: continue
        
        save_path = os.path.join(save_path_root, metric)
        pivot = metric_df.groupby(['Model', 'Scaling'])['Score'].mean().unstack()
        cols = [c for c in SCALERS_ORDER if c in pivot.columns]
        pivot = pivot[cols]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".3f", vmin=0, vmax=1)
        plt.title(f"Média Final Consolidada - {metric.upper()}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"summary_heatmap_{metric}.pdf"))
        plt.close()

def plot_granular_grids(df):
    """
    Gera grids 6x10 (Linhas=Scalers, Colunas=Datasets) para cada Modelo e Métrica.
    Isso permite ver a curva de Separação vs Score dataset por dataset.
    """
    print("\n--- Gerando Gráficos Granulares (Grids) ---")
    
    unique_models = df['Model'].unique()
    unique_datasets = sorted(df['Dataset_Num'].unique())
    
    # Divide datasets em chunks de 10 (ex: 1-10, 11-20...)
    dataset_chunks = [unique_datasets[i:i + 10] for i in range(0, len(unique_datasets), 10)]
    
    for model in unique_models:
        print(f"Processando Modelo: {model}...")
        
        for metric in METRICS:
            # Filtra dados para este modelo e métrica
            subset = df[(df['Model'] == model) & (df['Metric'] == metric)]
            if subset.empty:
                continue

            save_path = os.path.join(OUTPUT_DIR, "Granular_Grids", model, metric)
            os.makedirs(save_path, exist_ok=True)
            
            # Loop pelos grupos de 10 datasets
            for chunk in dataset_chunks:
                start_ds, end_ds = int(chunk[0]), int(chunk[-1])
                
                # Configura a figura (6 linhas x 10 colunas)
                fig, axes = plt.subplots(len(SCALERS_ORDER), len(chunk), 
                                         figsize=(20, 12), sharex=True, sharey=True)
                
                # Título Geral
                fig.suptitle(f"Modelo: {model} | Métrica: {metric.upper()} | Datasets: {start_ds}-{end_ds}", fontsize=16)
                
                # Preenche o Grid
                for row_idx, scaler in enumerate(SCALERS_ORDER):
                    for col_idx, ds_num in enumerate(chunk):
                        ax = axes[row_idx, col_idx]
                        
                        # Filtra dados específicos (Agrupa folds pela média)
                        data = subset[(subset['Dataset_Num'] == ds_num) & 
                                      (subset['Scaling'] == scaler)]
                        
                        # Agrupa por separação para ter a linha média
                        line_data = data.groupby('Separation')['Score'].mean().reset_index()
                        
                        if not line_data.empty:
                            ax.plot(line_data['Separation'], line_data['Score'], marker='.', linewidth=1)
                            ax.set_ylim(0, 1.1)
                            ax.grid(True, linestyle=':', alpha=0.6)
                        
                        # Rótulos (Apenas nas bordas para limpar o visual)
                        if row_idx == 0:
                            ax.set_title(f"DS {int(ds_num)}", fontsize=10)
                        if col_idx == 0:
                            ax.set_ylabel(scaler, fontsize=11, fontweight='bold')
                        
                # Rótulos globais de eixos
                fig.text(0.5, 0.01, 'Nível de Separação (Overlap)', ha='center', fontsize=12)
                fig.text(0.01, 0.5, 'Score (Média)', va='center', rotation='vertical', fontsize=12)
                
                plt.tight_layout(rect=[0.02, 0.02, 1, 0.96])
                filename = f"grid_{model}_{metric}_ds_{start_ds}-{end_ds}.pdf"
                plt.savefig(os.path.join(save_path, filename), dpi=100)
                plt.close(fig)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    df = load_and_consolidate_data()
    if df is not None:
        #plot_global_averages(df)
        plot_granular_grids(df)
        print("\nProcesso concluído! Verifique a pasta 'plots_granular'.")

if __name__ == "__main__":
    main()