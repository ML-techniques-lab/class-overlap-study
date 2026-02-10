import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import argparse

# --- Configurações ---
RESULTS_DIR = "results_2"  # Pasta com os CSVs
OUTPUT_DIR = "plots_granular"
SCALERS_ORDER = ['original', 'SS', 'MA', 'RS', 'QT', 'PT']
METRICS = ['accuracy', 'f1_score', 'g_mean', 'roc_auc']

def load_and_consolidate_data():
    """Lê todos os CSVs e converte para um formato longo (Tidy Data) com Overlap."""
    all_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    if not all_files:
        print(f"Nenhum arquivo encontrado em {RESULTS_DIR}")
        return None

    print(f"Lendo {len(all_files)} arquivos CSV...")
    data_list = []
    
    for file in all_files:
        try:
            # Lê o cabeçalho multinível
            df = pd.read_csv(file, header=[0, 1, 2, 3, 4], index_col=0)
            
            # Melt para transformar colunas em linhas
            df_melted = df.melt(ignore_index=False).reset_index()
            df_melted.columns = ['Fold', 'Model', 'Dataset', 'Separation', 'Scaling', 'Metric', 'Score']
            
            # Conversões
            df_melted['Separation'] = pd.to_numeric(df_melted['Separation'], errors='coerce')
            df_melted['Score'] = pd.to_numeric(df_melted['Score'], errors='coerce')
            df_melted['Dataset_Num'] = df_melted['Dataset'].str.extract('(\d+)').astype(float)
            
            # --- TRANSFORMAÇÃO: Overlap = 5 - Separação ---
            # 0 (Fácil/Separado) -> 5 (Difícil/Sobreposto)
            df_melted['Overlap'] = 5 - df_melted['Separation']
            
            data_list.append(df_melted)
        except Exception as e:
            print(f"Erro ao ler {file}: {e}")

    if not data_list:
        return None
        
    print("Consolidando dados...")
    return pd.concat(data_list, ignore_index=True)

def plot_global_averages(df):
    """Gera gráficos de LINHAS (Média) comparando performance vs OVERLAP."""
    print("\n--- Gerando Gráficos de Média Global (por Overlap) ---")
    save_path_root = os.path.join(OUTPUT_DIR, "Global_Averages")
    
    for metric in METRICS:
        metric_df = df[df['Metric'] == metric]
        if metric_df.empty: continue
            
        metric_path = os.path.join(save_path_root, metric)
        os.makedirs(metric_path, exist_ok=True)
        
        unique_models = metric_df['Model'].unique()
        for model in unique_models:
            model_data = metric_df[metric_df['Model'] == model]
            plt.figure(figsize=(10, 6))
            
            # Alterado eixo X para Overlap
            sns.lineplot(
                data=model_data, x='Overlap', y='Score', hue='Scaling', 
                hue_order=SCALERS_ORDER, marker='o', linewidth=2, palette="viridis"
            )
            
            plt.title(f"Média Geral: {model} - {metric.upper()}")
            plt.xlabel("Nível de Sobreposição (Overlap)") # 0 = Fácil, 5 = Difícil
            plt.ylim(0, 1.05)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(metric_path, f"avg_lineplot_{model}_{metric}.pdf"))
            plt.close()

def plot_granular_grids(df):
    """Gera grids detalhados (Datasets vs Scalers) usando Overlap no eixo X."""
    print("\n--- Gerando Gráficos Granulares ---")
    unique_models = df['Model'].unique()
    unique_datasets = sorted(df['Dataset_Num'].unique())
    dataset_chunks = [unique_datasets[i:i + 10] for i in range(0, len(unique_datasets), 10)]
    
    for model in unique_models:
        for metric in METRICS:
            subset = df[(df['Model'] == model) & (df['Metric'] == metric)]
            if subset.empty: continue

            save_path = os.path.join(OUTPUT_DIR, "Granular_Grids", model, metric)
            os.makedirs(save_path, exist_ok=True)
            
            for chunk in dataset_chunks:
                start_ds, end_ds = int(chunk[0]), int(chunk[-1])
                fig, axes = plt.subplots(len(SCALERS_ORDER), len(chunk), figsize=(20, 12), sharex=True, sharey=True)
                fig.suptitle(f"{model} | {metric.upper()} | DS {start_ds}-{end_ds}", fontsize=16)
                
                for row_idx, scaler in enumerate(SCALERS_ORDER):
                    for col_idx, ds_num in enumerate(chunk):
                        ax = axes[row_idx, col_idx]
                        data = subset[(subset['Dataset_Num'] == ds_num) & (subset['Scaling'] == scaler)]
                        
                        # Agrupa por Overlap em vez de Separação
                        line_data = data.groupby('Overlap')['Score'].mean().reset_index()
                        
                        if not line_data.empty:
                            ax.plot(line_data['Overlap'], line_data['Score'], marker='.', linewidth=1)
                            ax.set_ylim(0, 1.1)
                            ax.grid(True, linestyle=':', alpha=0.6)
                        
                        if row_idx == 0: ax.set_title(f"DS {int(ds_num)}")
                        if col_idx == 0: ax.set_ylabel(scaler, fontweight='bold')
                        # Adiciona label X apenas na última linha
                        if row_idx == len(SCALERS_ORDER) - 1: ax.set_xlabel("Overlap")
                
                plt.tight_layout(rect=[0.02, 0.02, 1, 0.96])
                plt.savefig(os.path.join(save_path, f"grid_{model}_{metric}_ds_{start_ds}-{end_ds}.pdf"))
                plt.close(fig)

def plot_range_by_model(df):
    """
    Visualização 1: Range (Max - Min) por Modelo.
    (Esta visualização independe do eixo X, pois calcula o range total).
    """
    print("\n--- Gerando Gráficos de Range (Por Modelo/Scaler) ---")
    save_path = os.path.join(OUTPUT_DIR, "Range_Analysis", "By_Model")
    os.makedirs(save_path, exist_ok=True)

    for metric in METRICS:
        metric_df = df[df['Metric'] == metric]
        if metric_df.empty: continue

        for scaler in SCALERS_ORDER:
            scaler_data = metric_df[metric_df['Scaling'] == scaler]
            if scaler_data.empty: continue

            agg = scaler_data.groupby('Model')['Score'].agg(['min', 'max'])
            agg['range'] = agg['max'] - agg['min']
            agg = agg.reset_index()

            plt.figure(figsize=(10, 6))
            sns.barplot(data=agg, x='Model', y='range', palette="rocket")
            
            plt.title(f"Range de Performance ({metric.upper()}) - Scaler: {scaler}")
            plt.ylabel(f"Range (Max - Min)")
            plt.xlabel("Modelo")
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            filename = f"range_by_model_{metric}_{scaler}.pdf"
            plt.savefig(os.path.join(save_path, filename))
            plt.close()

def plot_range_vs_overlap(df):
    """
    Visualização 2: Range vs OVERLAP.
    Eixo X: Nível de Sobreposição (0 a 5).
    """
    print("\n--- Gerando Gráficos de Range vs Sobreposição ---")
    save_path_root = os.path.join(OUTPUT_DIR, "Range_Analysis", "Vs_Overlap")
    
    for metric in METRICS:
        metric_df = df[df['Metric'] == metric]
        if metric_df.empty: continue

        unique_models = metric_df['Model'].unique()

        for model in unique_models:
            for scaler in SCALERS_ORDER:
                subset = metric_df[(metric_df['Model'] == model) & (metric_df['Scaling'] == scaler)]
                if subset.empty: continue

                # Agrupar por OVERLAP
                agg = subset.groupby('Overlap')['Score'].agg(['min', 'max'])
                agg['range'] = agg['max'] - agg['min']
                agg = agg.reset_index()

                if agg.empty: continue

                path_model = os.path.join(save_path_root, metric, model)
                os.makedirs(path_model, exist_ok=True)

                plt.figure(figsize=(8, 5))
                plt.plot(agg['Overlap'], agg['range'], marker='o', linewidth=2, color='tab:red')
                
                plt.title(f"Range vs Sobreposição: {model} ({scaler}) - {metric.upper()}")
                plt.xlabel("Nível de Sobreposição (Overlap)")
                plt.ylabel("Range (Max - Min)")
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.ylim(bottom=0)

                plt.tight_layout()
                filename = f"range_vs_overlap_{scaler}.pdf"
                plt.savefig(os.path.join(path_model, filename))
                plt.close()

def main():
    parser = argparse.ArgumentParser(description="Gerador de Visualizações de Experimentos")
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['global', 'granular', 'ranges', 'all'], 
        default='all',
        help="'global': Médias; 'granular': Grids detalhados; 'ranges': Análise Min/Max; 'all': Tudo."
    )
    args = parser.parse_args()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    df = load_and_consolidate_data()
    if df is not None:
        if args.mode in ['global', 'all']:
            plot_global_averages(df)
            
        if args.mode in ['granular', 'all']:
            plot_granular_grids(df)

        if args.mode in ['ranges', 'all']:
            plot_range_by_model(df)
            plot_range_vs_overlap(df)
            
        print(f"\nConcluído! Modo: {args.mode}. Verifique a pasta '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()