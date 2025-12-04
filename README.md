# Estudo de Impacto da Normalização na Sobreposição de Classes

Este estudo se propõe a analisar os impactos da normalização na performance de classificadores binários em cenários de sobreposição de classes (`overlap`). O objetivo é isolar a variável de separabilidade das classes (`class_sep`) para entender como técnicas de escalonamento (`MinMax e Standard Scaler`) influenciam modelos baseados em distância.

Abaixo há o detalhamento das etapas e experimentos realizados.

## 1. Geração de Datasets Sintéticos

Para gerar datasets sintéticos de maneira controlada, utilizamos a função `make_classification` da biblioteca `scikit-learn`, seguindo os parâmetros estocásticos definidos em `generate_datasets.py` e descritos no arquivo parameters.txt (ex: número de features informativas, redundantes, clusters, etc.).

Foram gerados 100 configurações base de datasets (N_TESTS = 100), garantindo uma diversidade estatística suficiente para os experimentos.

## 2. Variação de Sobreposição (Overlap)

O script principal de geração, `generate_datasets.py`, é responsável por criar as variações de cada um dos 100 datasets base. As seguintes regras foram aplicadas:

•	Variação de `class_sep`: Para cada dataset, são geradas 10 versões com o parâmetro de separação variando de 0.1 (alta sobreposição) até 5.0 (bem separados).

•	Controle de Desbalanceamento: Diferente do estudo original de desbalanceamento, aqui o peso das classes (`weights`) é definido aleatoriamente (entre 0.4 e 0.6) para cada dataset base e mantido fixo em todas as suas variações de sobreposição. Isso garante que as mudanças na performance sejam atribuídas puramente à mudança na geometria da separação, e não à mudança na proporção de classes.

Ao final, são gerados arquivos de visualização em `plots/`, ilustrando via PCA a transição da nuvem de pontos conforme o class_sep aumenta.

## 3. Pré-processamento e Escalonamento

O módulo generate_scaled_datasets.py processa os arquivos brutos gerados na etapa anterior. Para cada variação de sobreposição, são geradas três versões dos dados:

1.	Original: Dados brutos sem alteração.

2.	MM (`MinMaxScaler`): Dados normalizados para o intervalo `[0, 1]`.

3.	SS (`StandardScaler`): Dados padronizados (média 0, desvio padrão 1).

A estrutura de pastas resultante segue o padrão:

```scaled_datasets/dataset_{id}/sep_{valor}/{metodo}.csv```

## 4. Treinamento e Avaliação

O script de treinamento paralelo (ex: run_training.py) percorre a estrutura de pastas gerada e submete os dados aos classificadores.

•	Modelos: KNN (`$k=5$`) e GLVQ (`Generalized Learning Vector Quantization`).

•	Compatibilidade: Foram aplicados patches para garantir compatibilidade do `sklearn-lvq` com versões recentes do numpy e scipy.

•	Métricas: Accuracy, F1-Score, G-Mean e ROC-AUC.

•	Validação: Stratified K-Fold (`k=5`).

Os resultados finais são consolidados em um arquivo CSV (`results_overlap.csv`), permitindo a análise comparativa da eficácia da normalização em diferentes níveis de dificuldade de separação.

