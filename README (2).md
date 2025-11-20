Esse estudo se propõe a análisar os impactos da normalização na performance de classificadores binários em casos de desbalanceamento entre as classes. Abaixo há o detalhamento dos experimentos realizados

# Geração de datasets sintêticos

Para gerar datasets sintêticos de maneira controlada usamos a função `make_classification` da biblioteca scikit-learn de acordo com os parâmetros descritos no arquivo `parameters.txt`.

rodando o arquivo `initial_plots.py` teremos como saída dois plots, um que descreve a diferença da escala de cada feature de cada dataset e outra com o PCA de cada dataset gerado.

No total foram gerados 100 datasets que serão base para os experimentos (a princípio todos balanceados).

# Alterando os níveis de desbalanceamento

Rodando `generate_datasets.py` serão criados 10 copias de cada dataset base com os níveis de balanceamento em um intervalo entre 1 e aproximadamente 140. Há uma diferença entre o calculo do indice de desbalanceamento usado no artigo base e o peso da classe que a função aceita como parâmetro, esse calculo é detalhado no arquivo `generate_datasets.ipynb`.

Além disso os parâmetros base de cada dataset base é armazenado em um JSON e um plot do PCA dos 4 primeiros datasets são gerados mostrando a diferença entre cada nível de desbalanceamento.
