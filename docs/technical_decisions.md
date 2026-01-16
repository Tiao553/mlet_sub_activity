# Decisões Técnicas & Documentação de Experimentos

## 1. Visão Geral da Arquitetura

Este projeto implementa um pipeline robusto de Machine Learning para previsão de séries temporais de preços de ações. O sistema foi projetado para ser **model-agnostic** (agnóstico a modelo) e **parameter-driven** (guiado por parâmetros), permitindo que uma API selecione dinamicamente o melhor modelo pré-treinado com base no contexto de dados solicitado (Símbolo, Período e Intervalo).

### Componentes Chave

- **Executor de Experimentos (`models/run_grid_search.py`)**: Orquestra uma busca abrangente através de combinações de parâmetros (Grid Search).
- **Treinador Unificado (`models/train_model_grid.py`)**: Um script versátil que suporta tanto **TensorFlow** quanto **PyTorch**, lidando com pré-processamento de dados, treinamento e logging no MLflow.
- **Infraestrutura Dinâmica (`scripts/get_mlflow_uri.py`)**: Detecta automaticamente a localização do servidor de rastreamento do MLflow, suportando tanto desenvolvimento local quanto deployments na AWS EC2 sem alterações de código.
- **Motor de Relatórios (`scripts/generate_report.py`)**: Automatiza a seleção do melhor modelo e gera relatórios de justificativa legíveis por humanos.

## 2. Lógica de Seleção de Modelo

Suportamos dois frameworks primários de Deep Learning para aproveitar suas respectivas forças:

### TensorFlow (Keras)

- **Caso de Uso**: Prototipagem rápida e deployment em produção via formatos SavedModel padrão.
- **Arquiteturas**:
  - `LSTM`: Long Short-Term Memory padrão para capturar dependências temporais.
  - `GRU`: Gated Recurrent Unit, frequentemente mais rápido para treinar com performance similar.
  - `Bidirectional LSTM`: Captura contexto tanto do passado quanto do futuro (em sequências de treinamento), frequentemente resultando em melhor detecção de tendências.

### PyTorch

- **Caso de Uso**: Experimentos orientados a pesquisa e controle refinado sobre loops de treinamento.
- **Arquiteturas**: Implementa estruturas equivalentes de LSTM/GRU/BiLSTM para permitir comparação direta de performance com TensorFlow/Keras.

### Estratégia de Seleção "Switch Case"

A lógica de negócio central requer que a API sirva o melhor modelo para um `Período` específico (ex: 1 ano) e `Intervalo` (ex: 1 dia).

- Treinamos modelos para **todas as combinações válidas** dessas chaves de negócio.
- O script `promote_model.py` consulta o MLflow para encontrar a execução (run) com o **Menor RMSE** para cada tupla `(Símbolo, Período, Intervalo)`.
- Este "Vencedor" recebe a tag `@champion` e sua URI de Artefato é usada pela API para carregar o modelo.

## 3. Justificativa para Parâmetros Escolhidos

### Parâmetros de Dados

- **Tamanho da Sequência (Sequence Length - 24 vs 60)**:
  - *24*: Captura momemtum de curto prazo (ex: últimos 24 minutos/dias). Reage rapidamente à volatilidade.
  - *60*: Captura tendências de longo prazo (ex: última hora/bimestre), suavizando ruídos.
- **Conjunto de Features**:
  - *Completo*: Inclui Preço de Fechamento, MACD, RSI, Bandas de Bollinger e OBV. Empiricamente, descobrimos que incluir indicadores ponderados por volume (`OBV`, `VWAP`) melhora a precisão da predição para ações líquidas como VALE3 e PETR4.

### Parâmetros de Treinamento

- **Otimizador**: Adam é selecionado como padrão devido às suas propriedades de taxa de aprendizado adaptativa, que geralmente convergem mais rápido que SGD para RNNs.
- **Função de Perda (Loss Function)**: MSE (Mean Squared Error) é favorecido sobre MAE para treinamento para penalizar grandes erros (outliers) mais fortemente, o que é crítico na gestão de risco financeiro.

## 4. Acessando Artefatos

Todos os experimentos são logados no MLflow.

- **MLflow UI**: Acesso na porta 5000 (ex: `http://localhost:5000` ou `http://54.82.227.100:5000`).
- **Artefatos**:
  - **Modelos**: Armazenados como arquivos `.h5` (TF) ou `.pth` (Torch) dentro do diretório de artefatos da execução, hospedados no S3.
  - **Gráficos**: `prediction.png` mostra a visualização do Conjunto de Teste vs Predição para validação visual rápida.

## 5. Automação & DevOps

- **Resolução de IP**: O sistema usa `IMDSv2` para buscar IPs públicos da EC2 de forma segura. Isso previne hardcoding de IPs no código, permitindo deployments de infraestrutura imutável.
- **Git Sync**: O projeto usa sidecars `git-sync` no Docker para manter o código fresco nos servidores de orquestração (Airflow) sem necessidade de reconstruir imagens constantemente.

---
*Gerado pelo Agente Antigravity*
