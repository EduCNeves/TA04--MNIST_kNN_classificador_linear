
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def carregar_dados_teste():
    """Carrega o dataset completo e retorna apenas os conjuntos necessários para o teste final."""
    print("Carregando dados para avaliação final...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    imagens = np.concatenate([x_train, x_test], axis=0)
    rotulos = np.concatenate([y_train, y_test], axis=0)
    
    imagens_normalizadas = imagens / 255.0
    n_features = imagens.shape[1] * imagens.shape[2]
    imagens_achatadas = imagens_normalizadas.reshape(-1, n_features)

    X_treino_val, X_teste, y_treino_val, y_teste = train_test_split(
        imagens_achatadas, rotulos, test_size=0.15, random_state=42, stratify=rotulos
    )
    return X_treino_val, y_treino_val, X_teste, y_teste

if __name__ == "__main__":
    # Garante que a pasta de saída para os gráficos exista
    os.makedirs('output', exist_ok=True)
    
    # --- 1. Consolidação dos Resultados ---
    print("--- Lendo e consolidando resultados dos experimentos ---")
    df_knn = pd.read_csv('resultados_knn.csv')
    df_linear = pd.read_csv('resultados_linear.csv')
    df_geral = pd.concat([df_knn, df_linear], ignore_index=True)
    
    # --- 2. Geração dos Gráficos Comparativos ---
    print("--- Gerando gráficos comparativos ---")
    
    # Gráfico de Acurácia
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_geral, x='Abordagem', y='acuracia_valid', estimator=max, palette='viridis', hue='Abordagem', legend=False)
    plt.title('Acurácia Máxima na Validação por Abordagem', fontsize=16)
    plt.ylabel('Acurácia Máxima', fontsize=12); plt.xlabel('Abordagem Experimental', fontsize=12)
    plt.xticks(rotation=45, ha='right'); plt.ylim(0.8, 1.0); plt.tight_layout()
    plt.savefig('output/grafico_acuracia_maxima.png')
    plt.show()

    # --- 3. Seleção e Avaliação Final do Melhor Modelo ---
    print("\n--- Selecionando e avaliando o modelo campeão ---")
    
    # Encontra a melhor abordagem e seus parâmetros
    melhor_linha_geral = df_geral.loc[df_geral['acuracia_valid'].idxmax()]
    melhor_abordagem = melhor_linha_geral['Abordagem']
    
    print(f"Melhor abordagem encontrada: {melhor_abordagem} com acurácia de {melhor_linha_geral['acuracia_valid']:.4f}")
    
    # Carrega os dados para o teste final
    X_treino_val, y_treino_val, X_teste, y_teste = carregar_dados_teste()
    
    # Recria o melhor modelo
    modelo_final = None
    
    if 'kNN' in melhor_abordagem:
        # Encontra a melhor linha específica para essa abordagem kNN
        melhor_linha_especifica = df_knn.loc[df_knn[df_knn['Abordagem'] == melhor_abordagem]['acuracia_valid'].idxmax()]
        k_otimo = melhor_linha_especifica['k']
        metrica_otima = melhor_linha_especifica['metrica']
        peso_otimo = melhor_linha_especifica['peso']
        print(f"Parâmetros ótimos: k={k_otimo}, metrica={metrica_otima}, peso={peso_otimo}")
        
        modelo_final = KNeighborsClassifier(n_neighbors=int(k_otimo), metric=metrica_otima, weights=peso_otimo, n_jobs=-1)
        
        if 'PCA' in melhor_abordagem:
            n_comp = float(melhor_abordagem.split(' ')[-1].replace(')', ''))
            pca = PCA(n_components=n_comp)
            X_treino_val = pca.fit_transform(X_treino_val)
            X_teste = pca.transform(X_teste)
            

    # Treina o modelo final com todos os dados de treino+validação
    modelo_final.fit(X_treino_val, y_treino_val)
    
    # Avalia no conjunto de teste
    predicoes_finais = modelo_final.predict(X_teste)
    acuracia_final = accuracy_score(y_teste, predicoes_finais)
    
    print(f"\nACURÁCIA FINAL DO MODELO CAMPEÃO NO TESTE: {acuracia_final:.4f}")
    print("\nRelatório de Classificação Detalhado:")
    print(classification_report(y_teste, predicoes_finais))
    
    # Matriz de Confusão Final
    mat_conf = confusion_matrix(y_teste, predicoes_finais)
    plt.figure(figsize=(10, 8))
    sns.heatmap(mat_conf, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão do Modelo Final Campeão', fontsize=16)
    plt.xlabel('Rótulo Previsto'); plt.ylabel('Rótulo Verdadeiro')
    plt.savefig('output/matriz_confusao_final.png')
    plt.show()
