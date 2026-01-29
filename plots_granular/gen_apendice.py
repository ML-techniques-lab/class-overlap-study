import os

MODELS = ['AdaBoost', 'Bagging', 'DT', 'GLQV', 'GNB', 'GP', 'KNN', 'KNORAE', 
          'KNORAU', 'LCA', 'LDA', 'LR', 'MCB', 'MLP', 'OLA', 'Percep', 
          'QDA', 'RF', 'SVM_lin', 'SVM_rbf', 'XGBoost']
METRICS = ['accuracy', 'f1_score', 'g_mean', 'roc_auc']
DATASET_INTERVALS = ["1-10", "11-20", "21-30", "31-40", "41-50", 
                     "51-60", "61-70", "71-80", "81-90", "91-100"]

# Configuração do Grid
COLUNAS = 3
MAX_POR_PAGINA = 12  # Resulta em um grid 3x3
LARGURA_CELL = "0.32"

def generate_appendix():
    output_file = "apendice_gerado.tex"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(r"\section{Apêndice - Resultados Detalhados}" + "\n\n")
        
        # --- SEÇÃO 1: MÉDIAS GLOBAIS ---
        f.write(r"\subsection{Médias Globais por Modelo e Métrica}" + "\n")
        
        for metric in METRICS:
            f.write(f"\\subsubsection{{Métricas Globais: {metric.replace('_', ' ').upper()}}}\n")
            
            # Filtra apenas os modelos que possuem o arquivo
            arquivos_validos = []
            for model in MODELS:
                path_check = os.path.join("Global_Averages", metric, f"avg_lineplot_{model}_{metric}.pdf")
                if os.path.exists(path_check):
                    arquivos_validos.append((model, metric))
            
            # Divide os modelos em grupos de 6 (Grid 3x2)
            for i in range(0, len(arquivos_validos), MAX_POR_PAGINA):
                grupo = arquivos_validos[i : i + MAX_POR_PAGINA]
                
                f.write(r"\begin{figure}[H]" + "\n")
                f.write(r"    \centering" + "\n")
                
                for idx, (m, mt) in enumerate(grupo):
                    f.write(f"    \\cellMedia{{{LARGURA_CELL}}}{{{m}}}{{{mt}}}")
                    # Se for a coluna da esquerda, adiciona um \hfill. Se for a da direita ou o último, quebra linha.
                    if (idx + 1) % COLUNAS != 0 and (idx + 1) != len(grupo):
                        f.write(r" \hfill " + "\n")
                    else:
                        f.write(r" \\ \vspace{1em} " + "\n")
                
                f.write(f"    \\caption{{Médias Globais: {metric} (Continuação)}}" + "\n")
                f.write(r"\end{figure}" + "\n\n")
            
            f.write(r"\clearpage" + "\n")

        # --- SEÇÃO 2: GRIDS GRANULARES ---
        f.write(r"\subsection{Grids Granulares (Dataset por Dataset)}" + "\n")
        
        for model in MODELS:
            for metric in METRICS:
                # Coleta intervalos existentes
                intervalos_validos = []
                for interval in DATASET_INTERVALS:
                    path_check = os.path.join("Granular_Grids", model, metric, f"grid_{model}_{metric}_ds_{interval}.pdf")
                    if os.path.exists(path_check):
                        intervalos_validos.append(interval)
                
                if intervalos_validos:
                    f.write(f"\\subsubsection{{Granularidade: {model} - {metric.replace('_', ' ').upper()}}}\n")
                    
                    for i in range(0, len(intervalos_validos), MAX_POR_PAGINA):
                        grupo = intervalos_validos[i : i + MAX_POR_PAGINA]
                        
                        f.write(r"\begin{figure}[H]" + "\n")
                        f.write(r"    \centering" + "\n")
                        
                        for idx, interval in enumerate(grupo):
                            f.write(f"    \\cellGrid{{{LARGURA_CELL}}}{{{model}}}{{{metric}}}{{{interval}}}")
                            if (idx + 1) % COLUNAS != 0 and (idx + 1) != len(grupo):
                                f.write(r" \hfill " + "\n")
                            else:
                                f.write(r" \\ \vspace{1em} " + "\n")
                        
                        f.write(f"    \\caption{{Detalhamento {model} ({metric}) - Intervalos {grupo[0]} a {grupo[-1]}.}}" + "\n")
                        f.write(r"\end{figure}" + "\n\n")
                    
                    f.write(r"\clearpage" + "\n")

    print(f"Sucesso! Arquivo '{output_file}' gerado com layouts em grid.")

if __name__ == "__main__":
    generate_appendix()