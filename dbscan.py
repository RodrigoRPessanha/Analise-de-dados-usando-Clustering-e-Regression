#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from processar_dados import carregar_dados, preparar_dados_para_clustering

def encontrar_parametros_dbscan(X, k=5):
    """
    Encontra o parâmetro eps ótimo para DBSCAN usando o gráfico do cotovelo de k-distância.
    
    Args:
        X: Dados de entrada padronizados
        k: Número de vizinhos para considerar (padrão: 5)
    
    Returns:
        distancias: Distâncias ordenadas para cada ponto ao seu k-ésimo vizinho mais próximo
        eps_otimo: Valor de eps estimado pelo ponto de inflexão na curva k-distância
    """
    # Calcular as distâncias aos k vizinhos mais próximos
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    distancias = neigh.kneighbors(X)[0]
    
    # Obter a k-ésima distância para cada ponto (a última coluna)
    k_dist = np.sort(distancias[:, -1])
    
    # Calcular a derivada para encontrar ponto de inflexão
    derivada = np.gradient(k_dist)
    
    # Encontrar valor potencial para eps (ponto de inflexão)
    # Procuramos onde a derivada muda significativamente
    derivada_suavizada = np.convolve(derivada, np.ones(5)/5, mode='valid')  # Suavização
    potencial_indice = np.argmax(derivada_suavizada) + 2  # +2 por causa da suavização
    eps_otimo = k_dist[potencial_indice]
    
    print(f"Valor de eps recomendado pela análise: {eps_otimo:.3f}")
    
    # Plotar o gráfico de k-distância
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(k_dist)), k_dist)
    plt.axhline(y=eps_otimo, color='r', linestyle='--', 
                label=f'eps recomendado: {eps_otimo:.3f}')
    plt.title('Gráfico de k-distância para Seleção de eps')
    plt.xlabel('Pontos')
    plt.ylabel(f'Distância ao {k}º vizinho mais próximo')
    plt.grid(True)
    plt.legend()
    plt.savefig('dbscan_k_distance.png')
    
    return k_dist, eps_otimo

def otimizar_parametros_dbscan(X, y_true):
    """
    Testa diferentes combinações de parâmetros para DBSCAN.
    
    Args:
        X: Dados de entrada padronizados
        y_true: Classes reais para avaliação
    
    Returns:
        melhores_params: Melhores parâmetros encontrados
        melhor_ari: Melhor ARI encontrado
    """
    # Intervalo de valores para testar
    eps_values = np.linspace(0.4, 2.0, 9)  # Valores mais baixos a moderados
    min_samples_values = [3, 4, 5, 8, 10]
    
    resultados = []
    
    print("\nTestando diferentes combinações de parâmetros para DBSCAN:")
    print("="*60)
    print(f"{'eps':<8} {'min_samples':<12} {'Clusters':<10} {'Ruído %':<10} {'ARI':<10} {'Silhueta':<10}")
    print("-"*60)
    
    melhor_ari = -1
    melhores_params = None
    
    # Testar combinações de parâmetros
    for eps in eps_values:
        for min_samples in min_samples_values:
            # Executar DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            # Contar clusters e ruído
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            ruido_percentual = n_noise / len(labels) * 100
            
            # Calcular métricas se houver pelo menos 2 clusters
            if n_clusters >= 2:
                # Calcular ARI
                ari = adjusted_rand_score(y_true, labels)
                
                # Verificar se é possível calcular silhueta (excluindo ruído)
                silhueta = None
                if n_clusters >= 2:
                    mask = labels != -1
                    if sum(mask) > 1:  # Pelo menos 2 pontos não-ruído
                        silhueta = silhouette_score(X[mask], labels[mask])
                
                # Armazenar resultados
                resultados.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'ruido_percentual': ruido_percentual,
                    'ari': ari,
                    'silhueta': silhueta
                })
                
                # Imprimir resultado
                print(f"{eps:<8.3f} {min_samples:<12d} {n_clusters:<10d} {ruido_percentual:<10.1f} "
                      f"{ari:<10.4f} {silhueta if silhueta is not None else 'N/A'}")
                
                # Atualizar melhores parâmetros se ARI melhorou
                if ari > melhor_ari:
                    melhor_ari = ari
                    melhores_params = {'eps': eps, 'min_samples': min_samples}
    
    if melhores_params:
        print("\nMelhores parâmetros encontrados:")
        print(f"eps: {melhores_params['eps']:.3f}, min_samples: {melhores_params['min_samples']}")
        print(f"ARI: {melhor_ari:.4f}")
    else:
        print("\nNenhuma combinação válida de parâmetros encontrada.")
    
    return melhores_params, melhor_ari

def executar_dbscan(X, eps, min_samples):
    """
    Executa o algoritmo DBSCAN com os parâmetros especificados.
    
    Args:
        X: Dados de entrada padronizados
        eps: A distância máxima entre dois pontos para serem considerados na mesma vizinhança
        min_samples: Número mínimo de pontos numa vizinhança para formar um cluster core
    
    Returns:
        modelo: Modelo DBSCAN treinado
        labels: Rótulos de cluster atribuídos a cada amostra
    """
    # Inicializa e treina o modelo DBSCAN
    modelo = DBSCAN(eps=eps, min_samples=min_samples)
    
    # Ajustar o modelo e obter rótulos de cluster
    labels = modelo.fit_predict(X)
    
    # Contar o número de clusters (excluindo ruído, que é -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"Número de clusters estimado: {n_clusters}")
    print(f"Número de pontos de ruído: {n_noise} ({n_noise/len(labels)*100:.2f}%)")
    
    return modelo, labels

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
    # Ruído é representado por -1, vai ser plotado em preto
    ruido_mask = labels == -1
    if np.any(ruido_mask):
        plt.scatter(X_pca[ruido_mask, 0], X_pca[ruido_mask, 1], c='black', marker='x', label='Ruído')
    
    # Plotar clusters reais (não ruído)
    for cluster_label in set(labels) - {-1}:
        plt.scatter(
            X_pca[labels == cluster_label, 0],
            X_pca[labels == cluster_label, 1],
            label=f'Cluster {cluster_label}'
        )
    
    plt.title('Clusters DBSCAN (PCA)')
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
    plt.savefig('DBSCAN_clusters_pca.png')

def avaliar_clustering(labels, y_true, X):
    """
    Avalia o desempenho do clustering comparando com as classes reais.
    
    Args:
        labels: Rótulos de cluster atribuídos pelo algoritmo
        y_true: Classes reais
        X: Dados de entrada padronizados
    
    Returns:
        ari: Índice Rand Ajustado (quão similar é o agrupamento às classes reais)
        silhouette: Pontuação de silhueta (se possível calcular)
    """
    # Verificar se há pelo menos 2 clusters (excluindo ruído) para calcular a silhueta
    unique_clusters = set(labels) - {-1}
    
    silhouette = None
    if len(unique_clusters) >= 2:
        # Criar uma máscara para excluir pontos de ruído
        mask = labels != -1
        if sum(mask) > 1:  # Garantir que há pelo menos 2 pontos não-ruído
            silhouette = silhouette_score(X[mask], labels[mask])
            print(f"Pontuação de Silhueta (excluindo ruído): {silhouette:.4f}")
    else:
        print("Pontuação de Silhueta não calculável (menos de 2 clusters ou pontos por cluster)")
    
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
    
    return ari, silhouette

if __name__ == "__main__":
    # Carregar e preparar os dados
    dados = carregar_dados()
    X, y_true = preparar_dados_para_clustering(dados)
    
    # Encontrar parâmetros ótimos para DBSCAN com análise de k-distância
    print("Analisando o gráfico k-distância para estimar parâmetros iniciais...")
    k_dist, eps_estimado = encontrar_parametros_dbscan(X)
    
    # Otimização mais abrangente dos parâmetros do DBSCAN
    print("\nRealizando otimização mais abrangente dos parâmetros...")
    melhores_params, melhor_ari = otimizar_parametros_dbscan(X, y_true)
    
    if melhores_params:
        # Executar DBSCAN com os melhores parâmetros
        print(f"\nExecutando DBSCAN com os parâmetros otimizados:")
        print(f"eps={melhores_params['eps']:.3f}, min_samples={melhores_params['min_samples']}...")
        dbscan_model, dbscan_labels = executar_dbscan(X, melhores_params['eps'], melhores_params['min_samples'])
        
        # Visualizar os clusters
        print("\nVisualizando os clusters...")
        visualizar_clusters(X, dbscan_labels, y_true)
        
        # Avaliar o desempenho do clustering
        print("\nAvaliando o desempenho do clustering...")
        ari, silhouette = avaliar_clustering(dbscan_labels, y_true, X)
    else:
        print("Não foi possível executar o DBSCAN com parâmetros otimizados.")
    
    print("\nAlgoritmo DBSCAN concluído com sucesso!")