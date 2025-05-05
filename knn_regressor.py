#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from processar_dados import carregar_dados, preparar_dados_para_regressao

def treinar_knn_regressor(X, y, n_neighbors_range=None):
    """
    Treina e otimiza o modelo de regressão com KNN.
    
    Args:
        X: Características padronizadas
        y: Valores alvo (teor alcoólico dos vinhos)
        n_neighbors_range: Intervalo de valores para o número de vizinhos
        
    Returns:
        modelo: O melhor modelo encontrado
        param_grid: Grid de parâmetros utilizados
        grid_search: Objeto GridSearchCV com os resultados da pesquisa
    """
    # Definir o intervalo de valores para o número de vizinhos
    if n_neighbors_range is None:
        # Se não for especificado, usar um intervalo padrão
        n_neighbors_range = list(range(1, 21))
    
    # Definir o grid de parâmetros
    param_grid = {
        'n_neighbors': n_neighbors_range,
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # p=1 para distância Manhattan, p=2 para distância Euclidiana
    }
    
    # Inicializar o modelo base
    knn_reg = KNeighborsRegressor()
    
    # Configurar e executar a busca em grade com validação cruzada
    grid_search = GridSearchCV(knn_reg, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)
    
    # Obter e imprimir os melhores parâmetros
    print("Melhores parâmetros para KNN Regressor:")
    print(grid_search.best_params_)
    print(f"Melhor score (MSE negativo): {grid_search.best_score_:.4f}")
    
    # Treinar o modelo com os melhores parâmetros
    best_model = grid_search.best_estimator_
    
    return best_model, param_grid, grid_search

def avaliar_modelo(modelo, X, y, nome_modelo="KNN"):
    """
    Avalia o modelo usando validação cruzada e métricas de regressão.
    
    Args:
        modelo: Modelo treinado
        X: Características padronizadas
        y: Valores alvo
        nome_modelo: Nome do modelo para impressão
    
    Returns:
        dict: Dicionário com as métricas de avaliação
    """
    # Validação cruzada para MSE
    cv_scores = cross_val_score(
        modelo, X, y, cv=5, scoring='neg_mean_squared_error'
    )
    mse_cv = -cv_scores.mean()
    
    # Validação cruzada para R²
    cv_scores_r2 = cross_val_score(
        modelo, X, y, cv=5, scoring='r2'
    )
    r2_cv = cv_scores_r2.mean()
    
    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treinar o modelo nos dados de treinamento
    modelo.fit(X_train, y_train)
    
    # Fazer previsões nos dados de teste
    y_pred = modelo.predict(X_test)
    
    # Calcular métricas de avaliação
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nAvaliação do modelo {nome_modelo}:")
    print(f"MSE (Validação Cruzada): {mse_cv:.4f}")
    print(f"R² (Validação Cruzada): {r2_cv:.4f}")
    print(f"MSE (Teste): {mse:.4f}")
    print(f"RMSE (Teste): {rmse:.4f}")
    print(f"R² (Teste): {r2:.4f}")
    
    # Visualizar previsões vs. valores reais
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('Valores Reais')
    plt.ylabel('Valores Previstos')
    plt.title(f'{nome_modelo} - Previsões vs Valores Reais')
    plt.grid(True)
    plt.savefig(f'{nome_modelo.lower().replace(" ", "_")}_predicao.png')
    
    # Retornar as métricas como um dicionário
    return {
        'nome': nome_modelo,
        'mse_cv': mse_cv,
        'r2_cv': r2_cv,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

def visualizar_efeito_k(X, y):
    """
    Visualiza o efeito da escolha de diferentes valores de k (número de vizinhos)
    no erro de regressão.
    
    Args:
        X: Características padronizadas
        y: Valores alvo
    """
    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Testar diferentes valores de k
    k_values = list(range(1, 31))
    mse_values = []
    r2_values = []
    
    for k in k_values:
        # Treinar modelo com k vizinhos
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        # Fazer previsões e calcular erro
        y_pred = knn.predict(X_test)
        mse_values.append(mean_squared_error(y_test, y_pred))
        r2_values.append(r2_score(y_test, y_pred))
    
    # Plotar resultados
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_values, mse_values, 'o-')
    plt.title('Efeito de k no MSE')
    plt.xlabel('Número de vizinhos (k)')
    plt.ylabel('Erro Quadrático Médio')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(k_values, r2_values, 'o-')
    plt.title('Efeito de k no R²')
    plt.xlabel('Número de vizinhos (k)')
    plt.ylabel('Coeficiente de Determinação (R²)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('knn_effect_of_k.png')
    
    # Encontrar o k ótimo
    best_k_mse = k_values[np.argmin(mse_values)]
    best_k_r2 = k_values[np.argmax(r2_values)]
    
    print(f"Melhor k com base no MSE: {best_k_mse}")
    print(f"Melhor k com base no R²: {best_k_r2}")

if __name__ == "__main__":
    # Carregar dados
    dados = carregar_dados()
    X, y = preparar_dados_para_regressao(dados)
    
    # Visualizar o efeito de diferentes valores de k
    print("Analisando o efeito de diferentes valores de k...")
    visualizar_efeito_k(X, y)
    
    # Treinar modelo KNN com otimização de hiperparâmetros
    print("\nTreinando modelo KNN Regressor...")
    modelo_knn, param_grid, grid_search = treinar_knn_regressor(X, y)
    
    # Avaliar modelo
    metricas = avaliar_modelo(modelo_knn, X, y, "KNN")
    
    print("\nAlgoritmo KNN Regressor concluído com sucesso!")