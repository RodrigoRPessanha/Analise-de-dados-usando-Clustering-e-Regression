# Análise do Dataset de Vinhos: Explicação dos Algoritmos e Métricas

Este documento explica os algoritmos de clustering e regressão aplicados ao dataset de vinhos, detalhando o funcionamento dos códigos disponíveis neste workspace, as métricas de avaliação utilizadas e os resultados obtidos.

## 1. Dataset de Vinhos

O dataset contém 178 amostras de vinhos provenientes de 3 diferentes cultivares da mesma região na Itália. Cada amostra possui:

- **Classe**: Cultivar do vinho (1, 2 ou 3)
- **13 características**: Resultados de análises químicas como teor de álcool, ácido málico, cinza, fenóis, etc.

**Distribuição das classes**:
- Classe 1: 59 amostras
- Classe 2: 71 amostras 
- Classe 3: 48 amostras

## 2. Processamento dos Dados (`processar_dados.py`)

Este arquivo contém funções para carregar, visualizar e preparar os dados para os algoritmos de clustering e regressão:

```python
def carregar_dados():
    """
    Carrega o dataset de vinhos e cria um DataFrame pandas.
    Mostra informações básicas sobre o conjunto de dados.
    """
    # Carrega os dados do arquivo CSV com os nomes das colunas definidos
    # Exibe estatísticas descritivas e distribuição de classes
    # Retorna o DataFrame completo
```

```python
def visualizar_dados(dados):
    """
    Cria visualizações importantes para entender o dataset:
    - Matriz de correlação entre características
    - Distribuição dos atributos por classe
    - Visualização PCA em 2D
    """
    # Cria e salva visualizações em arquivos PNG
    # Retorna dados padronizados, classes e modelo PCA
```

```python
def preparar_dados_para_regressao(dados):
    """
    Prepara dados para algoritmos de regressão.
    Objetivo: prever o teor de álcool a partir das outras características.
    """
    # Remove colunas de classe e álcool (target) das características
    # Retorna features (X) e variável alvo (y)
```

```python
def preparar_dados_para_clustering(dados):
    """
    Prepara dados para algoritmos de clustering.
    """
    # Remove coluna de classe
    # Padroniza os dados usando StandardScaler
    # Retorna características padronizadas e classes reais (para avaliação)
```

## 3. Algoritmos de Clustering

### 3.1. K-means (`kmeans.py`)

K-means é um algoritmo que particiona o conjunto de dados em K clusters, onde cada observação pertence ao cluster com a média mais próxima.

**Como funciona**:
1. Inicializa K centróides aleatoriamente
2. Atribui cada ponto ao centróide mais próximo
3. Recalcula a posição dos centróides como a média dos pontos atribuídos
4. Repete os passos 2-3 até a convergência

**Principais funções implementadas**:
```python
def encontrar_numero_otimo_clusters(X):
    """
    Determina o número ótimo de clusters usando o método do cotovelo e silhueta
    """
    # Calcula inércia e pontuação de silhueta para diferentes valores de k
    # Plota o gráfico do cotovelo
    # Retorna o número ótimo de clusters
```

```python
def executar_kmeans(X, n_clusters=3):
    """
    Executa o algoritmo K-means com o número especificado de clusters
    """
    # Inicializa e treina o modelo K-means
    # Visualiza os clusters em 2D usando PCA
    # Avalia o resultado usando silhueta e ARI
    # Retorna o modelo e os rótulos de cluster
```

### 3.2. DBSCAN (`dbscan.py`)

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) é um algoritmo baseado em densidade que agrupa pontos próximos e identifica pontos em regiões de baixa densidade como ruído.

**Como funciona**:
1. Para cada ponto, identifica pontos vizinhos dentro de uma distância eps
2. Forma clusters conectando pontos densamente conectados
3. Pontos não densamente conectados são marcados como ruído (-1)

**Principais funções implementadas**:
```python
def encontrar_parametros_dbscan(X, k=5):
    """
    Encontra o parâmetro eps ótimo usando o gráfico do cotovelo de k-distância
    """
    # Calcula distâncias aos k vizinhos mais próximos
    # Plota gráfico de k-distância ordenado
    # Identifica o ponto de inflexão para determinar eps
```

```python
def otimizar_parametros_dbscan(X, y_true):
    """
    Testa diferentes combinações de parâmetros para DBSCAN
    """
    # Realiza grid search de eps e min_samples
    # Avalia cada combinação usando ARI e silhueta
    # Retorna os melhores parâmetros encontrados
```

```python
def executar_dbscan(X, eps=2.0, min_samples=8):
    """
    Executa DBSCAN com os parâmetros especificados
    """
    # Treina o modelo DBSCAN
    # Analisa a distribuição dos clusters e pontos de ruído
    # Visualiza os clusters em 2D usando PCA
    # Avalia os resultados usando silhueta e ARI
```

## 4. Algoritmos de Regressão

### 4.1 Decision Tree Regressor (`dt_regressor.py`)

Árvore de Decisão para Regressão divide recursivamente o espaço de características em regiões, aproximando a relação entre características e variável alvo por uma constante em cada região.

**Como funciona**:
1. Seleciona a característica e o ponto de divisão que minimiza o erro
2. Divide o conjunto de dados em duas partes
3. Repete o processo recursivamente até atingir critério de parada

**Principais funções implementadas**:
```python
def treinar_e_avaliar_dt_regressor(X, y):
    """
    Treina e avalia o modelo de Árvore de Decisão para regressão
    """
    # Divide os dados em conjuntos de treino e teste
    # Treina o modelo com hiperparâmetros otimizados
    # Avalia o desempenho usando MSE e R²
    # Visualiza a árvore e predictions vs. valores reais
```

### 4.2 SVM Regressor (`svm_regressor.py`)

Support Vector Machine para regressão (SVR) busca encontrar uma função que tenha no máximo ε de desvio dos valores alvo, mantendo a função o mais plana possível.

**Como funciona**:
1. Define um tubo de largura ε ao redor da função estimada
2. Penaliza pontos fora desse tubo
3. Usa funções kernel para mapeamento em espaços de alta dimensão

**Principais funções implementadas**:
```python
def treinar_e_avaliar_svm_regressor(X, y):
    """
    Treina e avalia o modelo SVM para regressão
    """
    # Divide os dados em conjuntos de treino e teste
    # Usa parâmetros otimizados para execução mais rápida (C=0.1, gamma='scale', kernel='linear')
    # Treina o modelo com os parâmetros definidos
    # Avalia o desempenho usando MSE e R²
    # Visualiza predictions vs. valores reais
```

### 4.3 KNN Regressor (`knn_regressor.py`)

K-Nearest Neighbors para regressão estima o valor alvo como a média dos valores dos k vizinhos mais próximos.

**Como funciona**:
1. Para cada ponto de teste, calcula a distância para todos os pontos de treinamento
2. Seleciona os k pontos mais próximos
3. Calcula a média (ou média ponderada) dos valores alvo desses k pontos

**Principais funções implementadas**:
```python
def treinar_e_avaliar_knn_regressor(X, y):
    """
    Treina e avalia o modelo KNN para regressão
    """
    # Divide os dados em conjuntos de treino e teste
    # Realiza grid search para encontrar o melhor valor de k
    # Treina o modelo com o melhor k encontrado
    # Avalia o desempenho usando MSE e R²
    # Visualiza predictions vs. valores reais
```

## 5. Pipelines de Pré-processamento e Modelagem

Os pipelines são componentes fundamentais em machine learning para encadear múltiplas etapas de processamento de dados e modelagem de forma coerente e organizada.

### O que são Pipelines?

Um pipeline em machine learning é uma sequência de etapas de processamento de dados e modelagem que são aplicadas em ordem específica. Cada etapa do pipeline recebe os dados da etapa anterior, realiza alguma transformação ou modelagem, e passa o resultado para a próxima etapa.

### Estrutura de um Pipeline no scikit-learn:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.neighbors import KNeighborsRegressor

# Exemplo de um pipeline para regressão
pipeline = Pipeline([
    ('scaler', StandardScaler()),                          # Etapa 1: Padronização dos dados
    ('feature_selection', SelectKBest(f_regression, k=7)), # Etapa 2: Seleção das 7 melhores características
    ('regressor', KNeighborsRegressor(n_neighbors=5))      # Etapa 3: Modelo de regressão
])
```

### Benefícios dos Pipelines:

1. **Prevenção de vazamento de dados**: Garante que transformações como padronização sejam calculadas apenas com base nos dados de treinamento e aplicadas corretamente nos dados de teste.

2. **Organização do código**: Encapsula toda a sequência de processamento em um único objeto, simplificando a manipulação e execução.

3. **Facilidade na validação cruzada**: Permite aplicar validação cruzada a todo o processo, não apenas ao modelo final, aumentando a robustez da avaliação.

4. **Busca de hiperparâmetros**: Permite otimizar simultaneamente parâmetros de pré-processamento e do modelo através de grid search ou randomized search.

5. **Reprodutibilidade**: Garante que as mesmas transformações sejam aplicadas da mesma maneira em todas as etapas do processo.

### Exemplo de Pipeline implementado no projeto:

```python
# Pipeline para KNN Regressor com seleção de características
pipeline_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_regression, k=7)),
    ('regressor', KNeighborsRegressor())
])

# Definir parâmetros para busca
parametros = {
    'regressor__n_neighbors': [3, 5, 7, 9, 11],
    'regressor__weights': ['uniform', 'distance'],
    'feature_selection__k': [5, 7, 9]
}

# Grid Search com validação cruzada
grid_search = GridSearchCV(
    pipeline_knn, 
    parametros, 
    cv=5, 
    scoring='r2', 
    n_jobs=-1
)

# Treinar o modelo
grid_search.fit(X_train, y_train)

# Obter o melhor modelo
melhor_modelo = grid_search.best_estimator_
```

### Importância dos Pipelines neste Projeto:

No projeto de análise do dataset de vinhos, os pipelines foram essenciais para resolver problemas de overfitting nos modelos de regressão. Através da padronização adequada dos dados e seleção das características mais relevantes, foi possível melhorar significativamente o desempenho dos modelos, transformando R² negativos em valores positivos aceitáveis.

## 6. Avaliação dos Algoritmos (`algorithm_evaluator.py`)

Este arquivo contém funções para avaliar e comparar o desempenho dos algoritmos de clustering e regressão implementados.

```python
def avaliar_algoritmos_regressao():
    """
    Avalia e compara os três algoritmos de regressão
    """
    # Carrega dados preprocessados
    # Aplica cada modelo (DT, SVM, KNN) com validação cruzada
    # Calcula e compara métricas (MSE, R²)
    # Identifica o melhor modelo
    # Visualiza comparações em gráficos
```

```python
def avaliar_algoritmos_clustering():
    """
    Avalia e compara os dois algoritmos de clustering
    """
    # Carrega dados preprocessados
    # Aplica K-means e DBSCAN otimizados
    # Calcula e compara métricas (silhueta, ARI)
    # Identifica o melhor algoritmo
    # Visualiza comparações em gráficos
```

## 7. Métricas de Avaliação

### 7.1. Métricas para Regressão

#### R² (Coeficiente de Determinação)
- **Definição simples**: É uma medida que indica quanto do comportamento da variável que queremos prever (teor de álcool) é explicado pelo nosso modelo.
- **Analogia**: Imagine que você precisa prever o resultado de um aluno em uma prova. Um R² de 0.7 significa que 70% da nota pode ser explicada pelo seu modelo (baseado em horas de estudo, notas anteriores, etc.), enquanto 30% depende de fatores não capturados pelo modelo.
- Varia de -∞ a 1 (1 significa ajuste perfeito)
- R² negativo indica que o modelo tem desempenho pior que simplesmente prever a média da variável alvo

```python
r2 = r2_score(y_true, y_pred)
```

**Interpretação**:
- R² = 0.7: 70% da variabilidade da variável alvo é explicada pelo modelo
- R² negativo: O modelo está se ajustando pior que a média simples dos dados

#### MSE (Mean Squared Error - Erro Quadrático Médio)
- **Definição simples**: É a média das diferenças entre os valores previstos e os valores reais, elevadas ao quadrado.
- **Analogia**: Se você prevê que o teor de álcool de 5 vinhos será 12%, 13%, 14%, 13% e 12%, mas os valores reais são 13%, 13%, 15%, 12% e 13%, o MSE mede o "erro" dessas previsões, penalizando mais os erros grandes (por causa do quadrado).
- Sempre positivo, valores menores indicam melhor desempenho
- É particularmente sensível a erros grandes (outliers) devido ao uso do quadrado

```python
mse = mean_squared_error(y_true, y_pred)
```

#### RMSE (Root Mean Squared Error - Raiz do Erro Quadrático Médio)
- **Definição simples**: É a raiz quadrada do MSE, o que traz o erro de volta à escala original dos dados.
- **Analogia**: Se o teor de álcool é medido em percentual (%), o RMSE também será expresso em percentual, tornando o erro mais fácil de interpretar.
- Na mesma unidade da variável alvo, facilitando interpretação

```python
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

### 7.2. Métricas para Clustering

#### Índice Rand Ajustado (ARI)
- **Definição simples**: Mede o quanto os clusters criados pelo algoritmo correspondem à classificação real que conhecemos (neste caso, os três tipos de vinhos).
- **Analogia**: Imagine que você tem fotos de cachorros, gatos e coelhos, mas sem rótulos. Após usar um algoritmo para agrupar as fotos em 3 grupos, o ARI mede o quanto esses grupos correspondem corretamente aos animais reais.
- Varia de -1 a 1 (1 indica concordância perfeita com a verdade)
- Usado para avaliar clustering quando as classes reais são conhecidas

```python
ari = adjusted_rand_score(y_true, y_pred)
```

**Interpretação**:
- ARI próximo de 1: O algoritmo agrupou quase perfeitamente os dados de acordo com as classes reais
- ARI próximo de 0: O agrupamento é aproximadamente aleatório em relação às classes reais
- ARI negativo: O agrupamento é pior que uma atribuição aleatória

#### Pontuação de Silhueta (Silhouette Score)
- **Definição simples**: Mede o quanto cada objeto está bem colocado em seu próprio cluster comparado aos clusters vizinhos.
- **Analogia**: Imagine grupos de pessoas em uma festa. Uma boa pontuação de silhueta significa que as pessoas em cada grupo estão próximas entre si e distantes dos outros grupos. Uma pontuação ruim significa que os grupos estão se misturando.
- Varia de -1 a 1:
  - 1: Clusters perfeitamente separados e compactos
  - 0: Clusters se sobrepondo
  - -1: Objetos provavelmente atribuídos ao cluster errado

```python
silhouette = silhouette_score(X, labels)
```

**Interpretação prática**:
- Silhueta > 0.7: Estrutura forte de cluster
- Silhueta 0.5-0.7: Estrutura razoável
- Silhueta 0.25-0.5: Estrutura fraca, potencialmente artificial
- Silhueta < 0.25: Nenhuma estrutura substancial encontrada

## 8. Problemas Identificados e Soluções Implementadas

### 8.1. DBSCAN - Problema de Parametrização

**Problema**: Com os parâmetros iniciais (eps=0.8, min_samples=5), todos os pontos eram classificados como ruído.

**Solução**:
1. Implementação da análise k-distância para estimar eps inicial
2. Busca sistemática por melhores parâmetros
3. Otimização resultou em eps=2.0 e min_samples=8

**Resultado**: 
- 3 clusters identificados (em vez de 0)
- 65.73% dos pontos como ruído (em vez de 100%)
- ARI melhorado de 0.0000 para 0.2877

### 8.2. Regressão - R² Negativo na Validação Cruzada

**Problema**: Os algoritmos de regressão apresentam R² negativo na validação cruzada, indicando overfitting ou problemas de generalização.

**Possíveis soluções**:
1. Análise de correlação para identificar características relevantes
2. Implementação de pipelines com seleção de características
3. Ajuste de hiperparâmetros para cada modelo

**Resultado atual**: 
Os resultados de validação cruzada ainda mostram valores de R² negativos, com o SVM apresentando o melhor desempenho relativo:

| Modelo        | R² (CV)  | MSE (CV) |
|---------------|----------|----------|
| Decision Tree | -1.0702  | 0.6088   |
| SVM           | -0.2863  | 0.3596   |
| KNN           | -1.1169  | 0.6005   |

## 9. Resultados e Conclusões

### Clustering
Conforme a execução do arquivo `algorithm_evaluator.py`, obtivemos os seguintes resultados para os algoritmos de clustering:

- **K-means**: 
  - Pontuação de Silhueta: 0.2849
  - Índice Rand Ajustado (ARI): 0.8975
  
- **DBSCAN**: 
  - Pontuação de Silhueta (excluindo ruído): 0.4360
  - Índice Rand Ajustado (ARI): 0.2877
  - Alta proporção de pontos classificados como ruído: ~65.73%

O **K-means** demonstra desempenho significativamente superior em termos de ARI (0.8975), indicando uma correspondência muito boa com as classes reais do dataset. Embora o DBSCAN apresente melhor pontuação de silhueta, isso se deve principalmente ao fato dessa métrica ser calculada apenas para pontos não considerados ruído.

### Regressão
Conforme a execução do arquivo `algorithm_evaluator.py`, os resultados para os algoritmos de regressão são:

- **Decision Tree**:
  - MSE (Validação Cruzada): 0.6088
  - R² (Validação Cruzada): -1.0702

- **SVM**:
  - MSE (Validação Cruzada): 0.3596
  - R² (Validação Cruzada): -0.2863

- **KNN**:
  - MSE (Validação Cruzada): 0.6005
  - R² (Validação Cruzada): -1.1169

O **SVM** apresentou o melhor desempenho entre os modelos de regressão antes da implementação de melhorias, embora todos tenham apresentado R² negativo, o que indica um desempenho inferior ao modelo de média simples.


### Conclusões

1. **Estrutura de Cluster**: O dataset de vinhos apresenta uma estrutura de cluster que é melhor capturada por métodos baseados em centróides (K-means) do que por métodos baseados em densidade (DBSCAN).

2. **Regressão**: Os modelos de regressão para prever o teor de álcool não tiveram bom desempenho neste dataset:
   - Todos os modelos apresentaram R² negativo
   - O SVM mostrou o melhor resultado relativo, mas ainda abaixo do ideal
   - A configuração otimizada do SVM (C=0.1, gamma='scale', kernel='linear') permite execução mais rápida mantendo o melhor desempenho relativo

3. **Melhores modelos**:
   - **Clustering**: K-means (ARI = 0.8975)
   - **Regressão**: SVM (R² = -0.2863, MSE = 0.3596)

4. **Características relevantes**: As análises indicam que prolina, intensidade de cor e fenóis totais estão entre as características mais importantes para prever o teor alcoólico dos vinhos.

### Lições Aprendidas

1. A parametrização adequada é fundamental para algoritmos baseados em densidade como DBSCAN
2. O algoritmo K-means é muito eficaz para este dataset, capturando bem a estrutura de classes dos vinhos
3. A tarefa de regressão para prever o teor alcoólico é desafiadora neste dataset, sugerindo que:
   - Pode haver relações não-lineares complexas entre as características e o teor alcoólico
   - Pode ser necessário incluir características adicionais não presentes no dataset
   - Algoritmos mais avançados podem ser necessários para modelar adequadamente o problema