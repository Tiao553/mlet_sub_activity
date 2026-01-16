# 📚 Referência da API

## URL Base

- **Produção (AWS API Gateway)**: `https://mdrzylhavh.execute-api.us-east-1.amazonaws.com/prod/`
- **Ambiente EC2 (Direto)**: `http://54.82.227.100:8000`
- **Local**: `http://localhost:8000`

## Endpoints

### 1. Obter Dados de Ações e Predição

Este endpoint busca dados históricos do Yahoo Finance, salva-os no S3 (Data Lake) e gera previsões de preço futuro utilizando o modelo campeão atual (Deployado via MLflow com a tag `@champion`).

**URL** : `/stock-data-prediction`

**Método** : `GET`

**Parâmetros de Consulta (Query Parameters)** :

| Parâmetro | Tipo | Obrigatório | Padrão | Descrição |
| :--- | :--- | :--- | :--- | :--- |
| `symbol` | `string` | **Sim** | - | Símbolo da ação (ex: `PETR4.SA`, `VALE3.SA`, `AAPL`). |
| `interval` | `string` | Não | `1d` | Granularidade dos dados (ex: `1m`, `5m`, `1h`, `1d`, `1wk`). |
| `period` | `string` | Não | - | Intervalo histórico para download (ex: `1mo`, `1y`). *Opcional se as datas de início/fim forem fornecidas.* |
| `start_date` | `string` | Não | - | Data de início no formato `YYYY-MM-DD`. |
| `end_date` | `string` | Não | `Hoje` | Data de fim no formato `YYYY-MM-DD`. |
| `auto_adjust` | `boolean` | Não | `True` | Se deve ajustar preços para desdobramentos/dividendos. |

**Resposta de Sucesso** :

**Código** : `200 OK`

**Exemplo de Conteúdo** :

```json
{
  "symbol": "PETR4.SA",
  "dates": ["2023-10-01", "2023-10-02", "2023-10-03"],
  "real_prices": [30.50, 31.00, 31.10],
  "predictions": [30.60, 30.95, 31.25],
  "metrics": {
      "last_close": 31.10,
      "predicted_next": 31.50,
      "model_version": "v3"
  }
}
```

**Resposta de Erro** :

**Código** : `500 Internal Server Error`

**Exemplo de Conteúdo** :

```json
{
  "detail": "Erro ao buscar dados do Yahoo Finance: Ticker inválido ou erro de conexão."
}
```

### 2. Health Check

Verifica se a API está online.

**URL** : `/`

**Método** : `GET`

**Resposta** : `{"message": "Welcome to the Stock Prediction API"}` 或 similar.
