import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import argparse

# --- Configurações ---
RESULTS_DIR = "results_2"
OUTPUT_DIR = "plots_consolidated"
MODEL_PALETTE = "tab10" 
SCALERS_ORDER = ['original', 'SS', 'MA', 'RS', 'QT', 'PT']
METRICS = ['accuracy', 'f1_score', 'g_mean', 'roc_auc']

def load_data_with_transformation():
    """
    Carrega os dados e aplica a transformação de Sobreposição.
    Overlap = 5 - Separação
    """
    all_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    if not all_files:
        print("Nenhum arquivo encontrado.")
        return None
        
    data_list = []
    print("Carregando e transformando dados...")
    for file in all_files:
        try:
            df = pd.read_csv(file, header=[0, 1, 2, 3, 4], index_col=0)
            df_melted = df.melt(ignore_index=False).reset_index()
            df_melted.columns = ['Fold', 'Model', 'Dataset', 'Separation', 'Scaling', 'Metric', 'Score']
            
            # Converte para numérico
            df_melted['Separation'] = pd.to_numeric(df_melted['Separation'], errors='coerce')
            df_melted['Score'] = pd.to_numeric(df_melted['Score'], errors='coerce')
            
            # --- TRANSFORMAÇÃO MATEMÁTICA AQUI ---
            # Cria a coluna 'Overlap' subtraindo a separação de 5
            # 5 (Muito separado) vira 0 (Sem overlap)
            # 0 (Muito junto) vira 5 (Muito overlap)
            df_melted['Overlap'] = 5 - df_melted['Separation']
            
            data_list.append(df_melted)
        except Exception as e:
            print(f"Erro ao ler {file}: {e}")
            pass
            
    return pd.concat(data_list, ignore_index=True) if data_list else None

def plot_models_comparison_by_scaler(df):
    """
    Gera comparação de modelos facetada por Normalização.
    Eixo X: Overlap (0 a 5).
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("\nGerando plots consolidados (Eixo X = Sobreposição)...")
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    for metric in METRICS:
        print(f"Processando métrica: {metric}...")
        metric_df = df[df['Metric'] == metric]
        valid_scalers = [s for s in SCALERS_ORDER if s in metric_df['Scaling'].unique()]
        
        g = sns.FacetGrid(
            metric_df, 
            col="Scaling", 
            col_order=valid_scalers,
            col_wrap=3, 
            height=4, 
            aspect=1.2,
            sharey=True
        )
        
        # Agora usamos 'Overlap' no eixo X
        g.map_dataframe(
            sns.lineplot, 
            x="Overlap", 
            y="Score", 
            hue="Model", 
            style="Model",
            markers=True, 
            dashes=True,
            palette=MODEL_PALETTE,
            linewidth=2,
            alpha=0.8
        )
        
        g.set_titles("{col_name}")
        # Rótulo explicito para não haver confusão
        g.set_axis_labels("Nível de Sobreposição (Overlap)", f"Score ({metric})")
        g.add_legend(title="Modelo", adjust_subtitles=True)
        
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(f"Performance vs. Dificuldade - {metric.upper()}", fontsize=16)
        
        filename = f"consolidated_models_overlap_{metric}.pdf"
        g.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
        plt.close()
        
    print(f"Pronto! Gráficos salvos em '{OUTPUT_DIR}'.")

def plot_scalers_comparison_by_model(df):
    """
    Gera comparação de normalizações facetada por Modelo.
    Eixo X: Overlap (0 a 5).
    """
    print("\nGerando plots invertidos...")
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    for metric in METRICS:
        metric_df = df[df['Metric'] == metric]
        
        g = sns.FacetGrid(
            metric_df, 
            col="Model", 
            col_wrap=3, 
            height=4, 
            aspect=1.2,
            sharey=True
        )
        
        g.map_dataframe(
            sns.lineplot, 
            x="Overlap", 
            y="Score", 
            hue="Scaling", 
            hue_order=SCALERS_ORDER,
            marker="o",
            palette="viridis",
            linewidth=2
        )

        g.set_titles("{col_name}")
        g.set_axis_labels("Nível de Sobreposição (Overlap)", f"Score ({metric})")
        g.add_legend(title="Normalização")
        
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(f"Robustez da Normalização - {metric.upper()}", fontsize=16)
        
        filename = f"consolidated_scalers_overlap_{metric}.pdf"
        g.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plots Consolidados com Overlap")
    parser.add_argument('--view', type=str, choices=['compare_models', 'compare_scalers', 'all'], default='all')
    args = parser.parse_args()
    
    df = load_data_with_transformation()
    if df is not None:
        if args.view in ['compare_models', 'all']:
            plot_models_comparison_by_scaler(df)
        if args.view in ['compare_scalers', 'all']:
            plot_scalers_comparison_by_model(df)

if __name__ == "__main__":
    main()