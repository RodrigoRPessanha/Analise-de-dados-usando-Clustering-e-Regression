#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from processar_dados import carregar_dados, preparar_dados_para_regressao

def treinar_svm_regressor(X, y):
    """
    Treina o modelo de regressão com SVM usando parâmetros predefinidos.
    
    Args:
        X: Características padronizadas
        y: Valores alvo (teor alcoólico dos vinhos)
        
    Returns:
        modelo: O modelo treinado com os parâmetros específicos
    """
    
    # Inicializar o modelo com os parâmetros específicos
    svm_reg = SVR(kernel='linear', C=0.1, gamma='scale')
    
    # Treinar o modelo
    svm_reg.fit(X, y)
    
    print("Treinamento concluído")
    
    return svm_reg

def avaliar_modelo(modelo, X, y, nome_modelo="SVM"):
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

if __name__ == "__main__":
    # Carregar dados
    dados = carregar_dados()
    X, y = preparar_dados_para_regressao(dados)
    
    # Treinar modelo
    print("Treinando modelo SVM Regressor...")
    modelo_svm = treinar_svm_regressor(X, y)
    
    # Avaliar modelo
    metricas = avaliar_modelo(modelo_svm, X, y, "SVM")
    
    print("\nAlgoritmo SVM Regressor concluído com sucesso!")