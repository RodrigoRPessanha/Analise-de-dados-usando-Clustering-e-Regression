#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Carregando o conjunto de dados
def carregar_dados():
    # Definindo os nomes das colunas de acordo com a descrição do dataset
    nomes_colunas = ['classe', 'alcool', 'acido_malico', 'cinza', 'alcalinidade_cinza', 
                    'magnesio', 'fenois_totais', 'flavonoides', 'fenois_nao_flavonoides', 
                    'proantocianidinas', 'intensidade_cor', 'tonalidade', 'od280_od315', 'prolina']
    
    # Carrega os dados
    dados = pd.read_csv('wine/wine.data', names=nomes_colunas, header=None)
    
    # Mostrar informações básicas sobre o conjunto de dados
    print(f"Forma do conjunto de dados: {dados.shape}")
    print("\nInformação sobre o conjunto de dados:")
    print(dados.info())
    print("\nEstatísticas descritivas:")
    print(dados.describe())
    
    # Verificar a distribuição das classes
    print("\nDistribuição das classes:")
    print(dados['classe'].value_counts())
    
    return dados

def visualizar_dados(dados):
    # Matriz de correlação
    plt.figure(figsize=(12, 10))
    corr = dados.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title('Matriz de Correlação')
    plt.tight_layout()
    plt.savefig('matriz_correlacao.png')
    
    # Visualização da distribuição dos atributos por classe
    plt.figure(figsize=(15, 10))
    for i in range(1, 14):
        plt.subplot(4, 4, i)
        for classe in [1, 2, 3]:
            sns.kdeplot(dados[dados['classe'] == classe].iloc[:, i], label=f'Classe {classe}')
        plt.title(dados.columns[i])
        plt.legend()
    plt.tight_layout()
    plt.savefig('distribuicao_atributos_por_classe.png')
    
    # PCA para visualização em 2D
    X = dados.drop('classe', axis=1)
    y = dados['classe']
    
    # Padronizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar PCA para reduzir para 2 dimensões
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Criar um DataFrame para fácil plotagem
    pca_df = pd.DataFrame(data=X_pca, columns=['Componente 1', 'Componente 2'])
    pca_df['classe'] = y
    
    # Plotar os resultados do PCA
    plt.figure(figsize=(10, 8))
    for classe in [1, 2, 3]:
        indices = pca_df['classe'] == classe
        plt.scatter(pca_df.loc[indices, 'Componente 1'], 
                   pca_df.loc[indices, 'Componente 2'], 
                   label=f'Classe {classe}')
    plt.title('PCA do Dataset de Vinhos')
    plt.xlabel('Primeira Componente Principal')
    plt.ylabel('Segunda Componente Principal')
    plt.legend()
    plt.grid(True)
    plt.savefig('pca_visualizacao.png')
    
    # Variância explicada por cada componente
    print(f"\nVariância explicada pelas duas primeiras componentes: {pca.explained_variance_ratio_.sum():.2f}")
    
    return X_scaled, y, pca

def preparar_dados_para_regressao(dados):
    """
    Prepara dados para algoritmos de regressão, usando a coluna "alcool" como alvo.
    Retorna os dados não padronizados, deixando a padronização para os pipelines.
    """
    # Para regressão, vamos prever o teor de álcool com base nas outras características
    X = dados.drop(['classe', 'alcool'], axis=1)
    y = dados['alcool']
    
    # Retornar os dados brutos para que os pipelines façam a padronização
    return X, y

def preparar_dados_para_clustering(dados):
    """
    Prepara dados para algoritmos de clustering, removendo a classe.
    """
    # Remover a coluna de classe para clustering não supervisionado
    X = dados.drop('classe', axis=1)
    y_true = dados['classe']  # Manter as classes verdadeiras para avaliação
    
    # Padronizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_true

if __name__ == "__main__":
    dados = carregar_dados()
    X_scaled, y, pca = visualizar_dados(dados)
    
    # Preparar dados para diferentes algoritmos
    X_reg, y_reg = preparar_dados_para_regressao(dados)
    X_cluster, y_cluster = preparar_dados_para_clustering(dados)
    
    print("\nDados processados e visualizados com sucesso!")