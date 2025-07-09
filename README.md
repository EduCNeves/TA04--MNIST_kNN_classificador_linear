# Projeto de Classificação de Dígitos MNIST

Este projeto explora e compara diferentes algoritmos de Machine Learning para a tarefa de classificação de dígitos manuscritos do dataset MNIST. Foram implementados e avaliados os classificadores k-Nearest Neighbors (kNN) e modelos lineares (SGDClassifier), investigando também o impacto da técnica de redução de dimensionalidade PCA.

O relatório final completo em formato de artigo pode ser encontrado no arquivo ` TA04__MNIST_kNN_classificador_linear.pdf`.

## Estrutura do Repositório

O projeto está organizado em scripts modulares para separar a fase de experimentação da fase de análise.

-   `experimento_knn.py`: Script para executar todos os experimentos com o modelo kNN, tanto nos dados originais quanto com PCA. Salva os resultados em `resultados_knn.csv`.
-   `experimento_linear.py`: Script para executar todos os experimentos com o modelo Linear (SGDClassifier), tanto nos dados originais quanto com PCA. Salva os resultados em `resultados_linear.csv`.
-   `gerar_relatorio_final.py`: Script principal que lê os resultados salvos, consolida os dados, gera os gráficos comparativos (salvando-os na pasta `output/`) e realiza a avaliação final do melhor modelo encontrado.
-   `requirements.txt`: Arquivo com as dependências do projeto para fácil instalação do ambiente.
-   ` TA04__MNIST_kNN_classificador_linear.pdf`: O relatório final do projeto em formato de artigo.

## Tecnologias e Bibliotecas

Este projeto foi desenvolvido em Python 3 e utiliza as seguintes bibliotecas principais:

-   **Scikit-learn:** Para a implementação dos modelos de Machine Learning (kNN, SGDClassifier, PCA) e métricas de avaliação.
-   **Pandas:** Para manipulação e análise dos dados dos experimentos.
-   **NumPy:** Para operações numéricas eficientes.
-   **TensorFlow (Keras):** Utilizado para o download e carregamento do dataset MNIST.
-   **Matplotlib & Seaborn:** Para a geração dos gráficos e visualizações.

## Como Executar

Para replicar os experimentos e gerar os resultados, siga os passos abaixo:

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/EduCNeves/TA04--MNIST_kNN_classificador_linear.git
    cd TA04--MNIST_kNN_classificador_linear
    ```

2.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Execute os scripts de experimentação:**
    * *Atenção: Estes scripts podem demorar alguns minutos para rodar, dependendo da sua máquina.*

    ```bash
    python experimento_knn.py
    python experimento_linear.py
    ```

4.  **Execute o script de análise e geração de relatório:**
    ```bash
    python gerar_relatorio_final.py
    ```
    * Após a execução, uma nova pasta chamada `output/` será criada, contendo os gráficos gerados. O console exibirá a análise final do modelo campeão.

## Resultados

A exploração sistemática demonstrou a alta performance do classificador kNN, especialmente quando combinado com a redução de dimensionalidade.

-   A configuração campeã foi o **kNN com k=3, aplicado sobre dados reduzidos pelo PCA para 87 dimensões** (90% da variância retida).
-   Este modelo alcançou uma robusta acurácia final de **97,54%** no conjunto de teste.
-   A aplicação do PCA provou ser extremamente benéfica, reduzindo o tempo de execução do kNN em mais de 95%, sem comprometer a acurácia.

## Autores

-   **Eduardo Camargo Neves** - GRR20196066
