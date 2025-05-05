#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from processar_dados import carregar_dados, preparar_dados_para_regressao

def treinar_decision_tree(X, y, max_depth_range=[None, 3, 5, 7, 10, 15]):
    """
    Treina e otimiza o modelo de regressão com Árvore de Decisão.
    
    Args:
        X: Características padronizadas
        y: Valores alvo (teor alcoólico dos vinhos)
        max_depth_range: Intervalo de valores para a profundidade máxima da árvore
        
    Returns:
        modelo: O melhor modelo encontrado
        param_grid: Grid de parâmetros utilizados
        grid_search: Objeto GridSearchCV com os resultados da pesquisa
    """
    # Definir o grid de parâmetros
    param_grid = {
        'max_depth': max_depth_range,
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Inicializar o modelo base
    dt_reg = DecisionTreeRegressor(random_state=42)
    
    # Configurar e executar a busca em grade com validação cruzada
    grid_search = GridSearchCV(dt_reg, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)
    
    # Obter e imprimir os melhores parâmetros
    print("Melhores parâmetros para Decision Tree Regressor:")
    print(grid_search.best_params_)
    print(f"Melhor score (MSE negativo): {grid_search.best_score_:.4f}")
    
    # Treinar o modelo com os melhores parâmetros
    best_model = grid_search.best_estimator_
    
    return best_model, param_grid, grid_search

def avaliar_modelo(modelo, X, y, nome_modelo="Decision Tree"):
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

def visualizar_importancia_caracteristicas(modelo, nomes_colunas):
    """
    Visualiza a importância das características para o modelo de árvore de decisão.
    
    Args:
        modelo: Modelo de árvore de decisão treinado
        nomes_colunas: Nomes das características
    """
    if not hasattr(modelo, 'feature_importances_'):
        print("O modelo não tem o atributo feature_importances_.")
        return
    
    # Obter a importância das características
    importancias = modelo.feature_importances_
    
    # Criar um DataFrame para facilitar a visualização
    indices = np.argsort(importancias)[::-1]
    df_importancias = pd.DataFrame({
        'Característica': [nomes_colunas[i] for i in indices],
        'Importância': importancias[indices]
    })
    
    print("\nImportância das Características:")
    print(df_importancias)
    
    # Visualizar graficamente
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(importancias)), importancias[indices], align='center')
    plt.xticks(range(len(importancias)), [nomes_colunas[i] for i in indices], rotation=90)
    plt.title('Importância das Características no Modelo Decision Tree')
    plt.tight_layout()
    plt.savefig('dt_feature_importance.png')

if __name__ == "__main__":
    # Carregar dados
    dados = carregar_dados()
    X, y = preparar_dados_para_regressao(dados)
    
    # Obter nomes das características para visualização posterior
    nomes_colunas = dados.drop(['classe', 'alcool'], axis=1).columns.tolist()
    
    # Treinar modelo
    print("Treinando modelo Decision Tree Regressor...")
    modelo_dt, param_grid, grid_search = treinar_decision_tree(X, y)
    
    # Avaliar modelo
    metricas = avaliar_modelo(modelo_dt, X, y, "Decision Tree")
    
    # Visualizar importância das características
    visualizar_importancia_caracteristicas(modelo_dt, nomes_colunas)
    
    print("\nAlgoritmo Decision Tree Regressor concluído com sucesso!")