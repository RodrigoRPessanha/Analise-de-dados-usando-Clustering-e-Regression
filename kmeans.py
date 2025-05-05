#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from processar_dados import carregar_dados, preparar_dados_para_clustering

def executar_kmeans(X, n_clusters=3):
    """
    Executa o algoritmo K-means com o número definido de clusters.
    
    Args:
        X: Dados de entrada padronizados
        n_clusters: Número de clusters (padrão: 3, mesmo número de classes no dataset)
    
    Returns:
        modelo: Modelo K-means treinado
        labels: Rótulos de cluster atribuídos a cada amostra
    """
    # Inicializa e treina o modelo K-means
    modelo = KMeans(n_clusters=n_clusters, 
                   init='k-means++',  # Método de inicialização avançado
                   n_init=10,         # Número de vezes para executar o algoritmo com diferentes centróides iniciais
                   max_iter=300,      # Número máximo de iterações
                   random_state=42)   # Para reprodutibilidade
    
    # Treinar o modelo e obter os rótulos de cluster
    labels = modelo.fit_predict(X)
    
    return modelo, labels

def encontrar_numero_otimo_clusters(X, range_clusters=[2, 3, 4, 5, 6, 7, 8]):
    """
    Encontra o número ótimo de clusters usando o método do cotovelo e
    a pontuação de silhueta.
    
    Args:
        X: Dados de entrada padronizados
        range_clusters: Intervalo de número de clusters para testar
    
    Returns:
        inertias: Lista de valores de inércia para cada número de clusters
        silhouette_scores: Lista de pontuações de silhueta para cada número de clusters
    """
    inertias = []
    silhouette_scores = []
    
    for n_clusters in range_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        
        # Calcular a inércia (soma das distâncias quadradas dentro do cluster)
        inertias.append(kmeans.inertia_)
        
        # Calcular a pontuação de silhueta
        if n_clusters > 1:  # Pontuação de silhueta requer pelo menos 2 clusters
            silhouette_scores.append(silhouette_score(X, labels))
        else:
            silhouette_scores.append(0)
    
    # Plotar o gráfico do método do cotovelo
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range_clusters, inertias, marker='o')
    plt.title('Método do Cotovelo')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inércia')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range_clusters, silhouette_scores, marker='o')
    plt.title('Pontuação de Silhueta')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Pontuação')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('kmeans_elbow_silhouette.png')
    
    return inertias, silhouette_scores

def visualizar_clusters(X, labels, y_true=None):
    """
    Visualiza os clusters usando PCA para redução de dimensionalidade.
    
    Args:
        X: Dados de entrada padronizados
        labels: Rótulos de cluster atribuídos pelo algoritmo
        y_true: Classes reais (se disponíveis)
    """
    # Aplicar PCA para redução de dimensionalidade para visualização
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(12, 10))
    
    # Plotar os clusters
    plt.subplot(2, 1, 1)
    for cluster_label in np.unique(labels):
        plt.scatter(
            X_pca[labels == cluster_label, 0],
            X_pca[labels == cluster_label, 1],
            label=f'Cluster {cluster_label}'
        )
    plt.title('Clusters KMeans (PCA)')
    plt.xlabel('Primeira Componente Principal')
    plt.ylabel('Segunda Componente Principal')
    plt.legend()
    plt.grid(True)
    
    # Se as classes reais estiverem disponíveis, plotá-las também para comparação
    if y_true is not None:
        plt.subplot(2, 1, 2)
        for classe in np.unique(y_true):
            plt.scatter(
                X_pca[y_true == classe, 0],
                X_pca[y_true == classe, 1],
                label=f'Classe {classe}'
            )
        plt.title('Classes Reais (PCA)')
        plt.xlabel('Primeira Componente Principal')
        plt.ylabel('Segunda Componente Principal')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('K-means_clusters_pca.png')

def avaliar_clustering(labels, y_true):
    """
    Avalia o desempenho do clustering comparando com as classes reais.
    
    Args:
        labels: Rótulos de cluster atribuídos pelo algoritmo
        y_true: Classes reais
    
    Returns:
        ari: Índice Rand Ajustado (quão similar é o agrupamento às classes reais)
    """
    # Calcular Índice Rand Ajustado (ARI)
    ari = adjusted_rand_score(y_true, labels)
    print(f"Índice Rand Ajustado: {ari:.4f}")
    
    # Criar uma tabela de contingência para ver como os clusters correspondem às classes
    contingency_table = pd.crosstab(
        pd.Series(y_true, name='Classes Reais'),
        pd.Series(labels, name='Clusters')
    )
    print("\nTabela de Contingência:")
    print(contingency_table)
    
    return ari

if __name__ == "__main__":
    # Carregar e preparar os dados
    dados = carregar_dados()
    X_scaled, y_true = preparar_dados_para_clustering(dados)
    
    # Encontrar o número ótimo de clusters
    print("Encontrando o número ótimo de clusters...")
    inertias, silhouette_scores = encontrar_numero_otimo_clusters(X_scaled)
    
    # Número de clusters de acordo com o conhecimento do dataset (3 tipos de vinho)
    n_clusters = 3
    
    # Executar K-means com o número escolhido de clusters
    print(f"\nExecutando K-means com {n_clusters} clusters...")
    kmeans_model, kmeans_labels = executar_kmeans(X_scaled, n_clusters)
    
    # Visualizar os clusters
    print("\nVisualizando os clusters...")
    visualizar_clusters(X_scaled, kmeans_labels, y_true)
    
    # Avaliar o desempenho do clustering
    print("\nAvaliando o desempenho do clustering...")
    ari = avaliar_clustering(kmeans_labels, y_true)
    
    print("\nAlgoritmo K-means concluído com sucesso!")