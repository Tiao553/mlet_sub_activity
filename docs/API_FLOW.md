# 🔄 Fluxo de Uso da API

Este diagrama ilustra o fluxo ponta-a-ponta de uma requisição ao endpoint `/stock-data-prediction`, detalhando como os dados são buscados, processados e como as previsões são geradas utilizando a arquitetura AWS.

```mermaid
sequenceDiagram
    autonumber
    actor Cliente
    participant API as Aplicação FastAPI
    participant Fetcher as Serviço de Busca (Fetcher)
    participant YFinance as Yahoo Finance
    participant S3 as AWS S3 (Raw & Processed)
    participant Predict as Serviço de Predição
    participant ModelLoader as Carregador de Modelo (MLflow)
    
    Cliente->>API: GET /stock-data-prediction (símbolo, período, intervalo)
    activate API
    
    API->>Fetcher: fetch_and_save_s3(símbolo, período, intervalo)
    activate Fetcher
    
    Fetcher->>YFinance: Download de Dados Históricos
    activate YFinance
    YFinance-->>Fetcher: Dados Brutos de Ações
    deactivate YFinance
    
    Fetcher->>S3: Salvar Dados Brutos (JSON) na 'raw-zone'
    activate S3
    S3-->>Fetcher: Sucesso
    deactivate S3
    
    Fetcher-->>API: Status 200 (Sucesso na Busca)
    deactivate Fetcher
    
    API->>Predict: pipe_to_predict(símbolo, inicio, fim, período, intervalo)
    activate Predict
    
    Predict->>S3: Ler Dados Brutos
    activate S3
    S3-->>Predict: Dados JSON
    deactivate S3
    
    Predict->>Predict: Pré-processamento (Normalização, Criação de Sequências)
    
    Predict->>ModelLoader: Carregar Modelo Campeão (@champion)
    activate ModelLoader
    ModelLoader-->>Predict: Modelo PyTorch/Keras Carregado
    deactivate ModelLoader
    
    Predict->>Predict: Gerar Predição (Inferência)
    
    Predict-->>API: Resultado da Predição (JSON)
    deactivate Predict
    
    API-->>Cliente: Resposta {datas, preços_reais, predicoes, métricas}
    deactivate API
```

### Descrição dos Componentes

1. **Cliente**: Usuário final ou sistema externo que consome a API.
2. **Aplicação FastAPI**: Ponto de entrada (Gateway) que orquestra os pedidos.
3. **Serviço de Busca (Fetcher)**: Responsável por obter os dados mais recentes do Yahoo Finance para garantir que a predição use dados atualizados.
4. **AWS S3**: Armazenamento durável usado como Data Lake (camadas Raw e Trusted) e para artefatos do MLflow.
5. **Serviço de Predição**: Núcleo de inteligência que prepara os dados e executa o modelo.
6. **Carregador de Modelo**: Interage com o MLflow para baixar o modelo marcado como produção (`@champion`) dinamicamente.
