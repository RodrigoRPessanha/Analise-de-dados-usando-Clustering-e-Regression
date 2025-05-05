#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import cross_val_score
from processar_dados import carregar_dados, preparar_dados_para_regressao, preparar_dados_para_clustering

def avaliar_algoritmos_regressao():
    """
    Avalia e compara os três algoritmos de regressão linear:
    Decision Tree Regressor, SVM Regressor e KNN Regressor.
    """
    # Carregar os dados
    dados = carregar_dados()
    X, y = preparar_dados_para_regressao(dados)
    
    # Definir os modelos com parâmetros otimizados (baseados na análise prévia)
    modelos = {
        'Decision Tree': DecisionTreeRegressor(max_depth=5, min_samples_leaf=1, min_samples_split=2, random_state=42),
        'SVM': SVR(C=0.1, gamma='scale', kernel='linear'),  # Parâmetros atualizados para execução mais rápida
        'KNN': KNeighborsRegressor(n_neighbors=5, p=2, weights='distance')
    }
    
    # Métricas para armazenar os resultados
    resultados = pd.DataFrame(
        columns=['Modelo', 'MSE (Validação Cruzada)', 'R² (Validação Cruzada)'],
        index=range(len(modelos))
    )
    
    print("Avaliando algoritmos de regressão...")
    print("-" * 50)
    
    # Avaliar cada modelo
    for i, (nome, modelo) in enumerate(modelos.items()):
        # Calcular MSE com validação cruzada
        cv_scores_mse = cross_val_score(
            modelo, X, y, cv=5, scoring='neg_mean_squared_error'
        )
        mse_cv = -cv_scores_mse.mean()
        
        # Calcular R² com validação cruzada
        cv_scores_r2 = cross_val_score(
            modelo, X, y, cv=5, scoring='r2'
        )
        r2_cv = cv_scores_r2.mean()
        
        # Armazenar resultados
        resultados.iloc[i] = [nome, mse_cv, r2_cv]
        
        print(f"Modelo: {nome}")
        print(f"MSE (Validação Cruzada): {mse_cv:.4f}")
        print(f"R² (Validação Cruzada): {r2_cv:.4f}")
        print("-" * 50)
    
    # Encontrar o melhor modelo com base no R²
    melhor_r2_idx = resultados['R² (Validação Cruzada)'].idxmax()
    melhor_modelo_r2 = resultados.iloc[melhor_r2_idx]['Modelo']
    
    # Encontrar o melhor modelo com base no MSE
    melhor_mse_idx = resultados['MSE (Validação Cruzada)'].idxmin()
    melhor_modelo_mse = resultados.iloc[melhor_mse_idx]['Modelo']
    
    print("\nResultado da avaliação dos algoritmos de regressão:")
    print(f"Melhor modelo baseado em R²: {melhor_modelo_r2}")
    print(f"Melhor modelo baseado em MSE: {melhor_modelo_mse}")
    
    # Visualizar comparação dos modelos
    plt.figure(figsize=(12, 6))
    
    # Gráfico para MSE
    plt.subplot(1, 2, 1)
    plt.bar(resultados['Modelo'], resultados['MSE (Validação Cruzada)'])
    plt.title('MSE por Modelo')
    plt.ylabel('MSE (menor é melhor)')
    plt.xticks(rotation=15)
    plt.grid(axis='y')
    
    # Gráfico para R²
    plt.subplot(1, 2, 2)
    plt.bar(resultados['Modelo'], resultados['R² (Validação Cruzada)'])
    plt.title('R² por Modelo')
    plt.ylabel('R² (maior é melhor)')
    plt.xticks(rotation=15)
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig('comparacao_regressores.png')
    
    return resultados, melhor_modelo_r2, melhor_modelo_mse

def avaliar_algoritmos_clustering():
    """
    Avalia e compara os dois algoritmos de clustering:
    K-means e DBSCAN.
    """
    # Carregar os dados
    dados = carregar_dados()
    X, y_true = preparar_dados_para_clustering(dados)
    
    print("Avaliando algoritmos de clustering...")
    print("-" * 50)
    
    # Aplicar K-means
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    
    # Calcular métricas para K-means
    kmeans_silhouette = silhouette_score(X, kmeans_labels)
    kmeans_ari = adjusted_rand_score(y_true, kmeans_labels)
    
    print("K-means:")
    print(f"Pontuação de Silhueta: {kmeans_silhouette:.4f}")
    print(f"Índice Rand Ajustado (ARI): {kmeans_ari:.4f}")
    print("-" * 50)
    
    # Aplicar DBSCAN com parâmetros otimizados
    dbscan = DBSCAN(eps=2.0, min_samples=8)  # Parâmetros otimizados
    dbscan_labels = dbscan.fit_predict(X)
    
    # Calcular métricas para DBSCAN
    # Verificar se há pelo menos 2 clusters (excluindo ruído) para calcular silhueta
    unique_clusters = set(dbscan_labels) - {-1}
    dbscan_silhouette = None
    if len(unique_clusters) >= 2:
        # Criar uma máscara para excluir pontos de ruído
        mask = dbscan_labels != -1
        if sum(mask) > 1:  # Garantir que há pelo menos 2 pontos não-ruído
            dbscan_silhouette = silhouette_score(X[mask], dbscan_labels[mask])
    
    dbscan_ari = adjusted_rand_score(y_true, dbscan_labels)
    
    print("DBSCAN:")
    if dbscan_silhouette is not None:
        print(f"Pontuação de Silhueta (excluindo ruído): {dbscan_silhouette:.4f}")
    else:
        print("Pontuação de Silhueta: Não calculável (menos de 2 clusters ou pontos por cluster)")
    print(f"Índice Rand Ajustado (ARI): {dbscan_ari:.4f}")
    print("-" * 50)
    
    # Comparar resultados
    # print("\nResultado da avaliação dos algoritmos de clustering:")
    
    # Comparar com base no ARI (indica o quanto o clustering se alinha às classes reais)
    if kmeans_ari > dbscan_ari:
        melhor_algoritmo_ari = "K-means"
    else:
        melhor_algoritmo_ari = "DBSCAN"
        
    result_ari = f"Melhor algoritmo baseado no Índice Rand Ajustado: {melhor_algoritmo_ari}"
    
    # Se possível, comparar com base na silhueta também
    if dbscan_silhouette is not None:
        if kmeans_silhouette > dbscan_silhouette:
            melhor_algoritmo_silhouette = "K-means"
        else:
            melhor_algoritmo_silhouette = "DBSCAN"
        result_silhouette = f"Melhor algoritmo baseado na Pontuação de Silhueta: {melhor_algoritmo_silhouette}"
    
    # Visualizar comparação dos algoritmos (apenas para ARI, já que silhouette pode não estar disponível para DBSCAN)
    plt.figure(figsize=(10, 6))
    algoritmos = ['K-means', 'DBSCAN']
    ari_scores = [kmeans_ari, dbscan_ari]
    
    plt.bar(algoritmos, ari_scores)
    plt.title('Comparação de Algoritmos de Clustering - Índice Rand Ajustado')
    plt.ylabel('ARI (maior é melhor)')
    plt.ylim(0, 1)  # ARI está entre -1 e 1, mas geralmente positivo para correspondências razoáveis
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig('comparacao_clustering.png')
    
    # Retornar resultados e melhor algoritmo
    resultados = {
        'K-means': {'silhouette': kmeans_silhouette, 'ari': kmeans_ari},
        'DBSCAN': {'silhouette': dbscan_silhouette, 'ari': dbscan_ari},
        'Resultados': [melhor_algoritmo_ari,melhor_algoritmo_silhouette]
    }
    
    return resultados, melhor_algoritmo_ari

if __name__ == "__main__":
    print("Avaliando algoritmos...")
    
    # Avaliar algoritmos de regressão
    print("\n*** AVALIAÇÃO DE ALGORITMOS DE REGRESSÃO ***\n")
    resultados_regressao, melhor_r2, melhor_mse = avaliar_algoritmos_regressao()
    
    # Avaliar algoritmos de clustering
    print("\n*** AVALIAÇÃO DE ALGORITMOS DE CLUSTERING ***\n")
    resultados_clustering, melhor_clustering = avaliar_algoritmos_clustering()

    
    
    print("\nResultados de Regressão:")
    print(f"Resultado com base no R²: {melhor_r2}")
    print(f"Resultado com base no MSE: {melhor_mse}")
    print("\nResultados de Clustering:")
    print(f"Resultado com base no ARI (indica o quanto o clustering se alinha às classes reais): {resultados_clustering['Resultados'][0]}")
    print(f"Resultado com base na Silhueta (se disponível): {resultados_clustering['Resultados'][1]}\n")
    print("\nAvaliação concluída com sucesso!")