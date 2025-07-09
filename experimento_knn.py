import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

def carregar_e_preparar_dados():
    """Carrega, pré-processa e divide o dataset MNIST."""
    print("Carregando e preparando os dados...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    imagens = np.concatenate([x_train, x_test], axis=0)
    rotulos = np.concatenate([y_train, y_test], axis=0)
    
    imagens_normalizadas = imagens / 255.0
    n_features = imagens.shape[1] * imagens.shape[2]
    imagens_achatadas = imagens_normalizadas.reshape(-1, n_features)

    X_treino, X_temp, y_treino, y_temp = train_test_split(
        imagens_achatadas, rotulos, train_size=0.7, random_state=42, stratify=rotulos
    )
    X_valid, _, y_valid, _ = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    return X_treino, y_treino, X_valid, y_valid

def rodar_experimentos_knn(X_treino, y_treino, X_valid, y_valid, abordagem_base):
    """Roda um conjunto de experimentos para o kNN e retorna um DataFrame."""
    k_candidatos = [1, 3, 5, 9, 13, 15, 21]
    metricas_distancia = ['euclidean', 'manhattan', 'chebyshev']
    pesos_vizinhos = ['uniform', 'distance']
    resultados = []

    for metrica in metricas_distancia:
        for peso in pesos_vizinhos:
            for k in k_candidatos:
                print(f"Testando {abordagem_base}: k={k}, metrica={metrica}, peso={peso}")
                modelo_knn = KNeighborsClassifier(n_neighbors=k, metric=metrica, weights=peso, n_jobs=-1)
                modelo_knn.fit(X_treino, y_treino)
                predicoes = modelo_knn.predict(X_valid)
                resultados.append({
                    'Abordagem': abordagem_base,
                    'acuracia_valid': accuracy_score(y_valid, predicoes),
                    'k': k, 'metrica': metrica, 'peso': peso
                })
    return pd.DataFrame(resultados)

if __name__ == "__main__":
    X_treino, y_treino, X_valid, y_valid = carregar_e_preparar_dados()
    
    # 1. Experimentos com dados originais
    df_knn_original = rodar_experimentos_knn(X_treino, y_treino, X_valid, y_valid, 'kNN (Original)')
    
    lista_dfs = [df_knn_original]
    
    # 2. Experimentos com PCA
    for n_comp in [0.95, 0.90]:
        print(f"\n===== kNN com PCA {n_comp} =====\n")
        pca = PCA(n_components=n_comp)
        X_treino_pca = pca.fit_transform(X_treino)
        X_valid_pca = pca.transform(X_valid)
        
        df_knn_pca = rodar_experimentos_knn(X_treino_pca, y_treino, X_valid_pca, f'kNN (PCA {n_comp})')
        lista_dfs.append(df_knn_pca)
        
    # 3. Consolidar e salvar resultados
    df_resultados_finais_knn = pd.concat(lista_dfs, ignore_index=True)
    df_resultados_finais_knn.to_csv('resultados_knn.csv', index=False)
    
    print("\nExperimentos com kNN concluídos! Resultados salvos em 'resultados_knn.csv'")
