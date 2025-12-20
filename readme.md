# Atividade Substitutiva da Pós-Graduação MLET03 da FIAP

[link do video de apresentacao](https://youtu.be/v03U9tBDizg)

> **Projeto de Operacionalização de Modelo de Previsão de Ações**
>
> Este projeto é uma atividade substitutiva para a disciplina MLET03. O objetivo é operacionalizar um modelo de Deep Learning (LSTM) para previsão de preços de ações, integrando-o com **MLflow** para rastreamento de experimentos e gerenciamento de modelos, além de preparar a infraestrutura para deploy e serving.

---

## Sumário

- [Descrição](#descrição)
- [Objetivo](#objetivo)
- [Arquitetura do Projeto](#arquitetura-do-projeto)
- [Fases do Projeto](#fases-do-projeto)
- [Requisitos](#requisitos)
- [Instalação e Execução](#instalação-e-execução)
- [Uso da API](#uso-da-api)
- [Licença](#licença)

---

## 📘 Descrição

Este projeto utiliza uma rede neural **LSTM** (Long Short-Term Memory) para prever o valor de fechamento de ações (ex: VALE3.SA) no dia seguinte. A grande mudança neste cenário é a **operacionalização** do modelo, focando em:

1. **Rastreabilidade**: Uso do MLflow para registrar métricas, parâmetros e artefatos de treinamento.
2. **Reprodutibilidade**: Estruturação dos experimentos para garantir que os resultados possam ser reproduzidos.
3. **Model Serving**: Planejamento pa uso do MLflow como servidor de modelo ou integrado à API.

---

## 🎯 Objetivo

Criar uma solução robusta que não apenas treine um modelo, mas gerencie seu ciclo de vida:

- **Experimentação Controlada**: Salvar e comparar múltiplas execuções com diferentes hiperparâmetros no MLflow.
- **API de Inferência**: Disponibilizar o melhor modelo via API REST (FastAPI).
- **Infraestrutura**: Containerização com Docker e orquestração dos serviços (API + MLflow).

---

## 🧱 Arquitetura do Projeto

```sh
.
├── api                          # Código da aplicação (FastAPI)
│   ├── app                     # Lógica de aplicação
│   │   ├── api.py              # Endpoints da API
│   │   ├── config/logger.py    # Configuração de logs
│   │   ├── services/           # Serviços para predição e fetch
│   │   └── model/              # Modelos (será depreciado em favor do MLflow)
│   ├── dockerfile              # Dockerfile da API
├── infrastructure              # Infraestrutura (Terraform/Docker)
├── models                      # Scripts de treinamento e MLflowManager
│   ├── hiperparams_train_test_torch_seq.py
│   └── mlflow_manager.py       # (Novo) Gerenciador do MLflow
├── notebooks                   # Análises exploratórias
├── docker-compose.yml          # Orquestração (API + MLflow)
└── readme.md                   # Este arquivo
```

---

## � Fases do Projeto

1. **Setup e Documentação**: Ajuste do README e infraestrutura básica (MLflow).
2. **Gerenciador MLflow**: Criação de classe para abstrair o uso do MLflow.
3. **Experimentação**: Execução massiva de experimentos e seleção de modelos.
4. **Model Serving**: Integração do modelo via MLflow na API.
5. **API Gateway**: Ajustes finais e gateway.

---

## � Requisitos

- Python 3.10+
- Docker & Docker Compose
- Bibliotecas: PyTorch, MLflow, FastAPI, Pandas, Scikit-learn, Ta-Lib.

---

## 🚀 Instalação e Execução

### Usando Docker Compose (Recomendado)

O ambiente inclui a API e o servidor MLflow.

```bash
# Subir os serviços
docker-compose up -d --build
```

- **API**: <http://localhost:8000>
- **MLflow UI**: <http://localhost:5000>

---

## 📡 Uso da API

Endpoint: `/stock-data-prediction` (GET)

**Parâmetros:**

- `symbol`: Código da ação (ex: `VALE3.SA`).
- `period`: Período histórico (ex: `7d`, `1mo`).
- `interval`: Intervalo (ex: `1m`, `1d`).

**Exemplo:**

```bash
curl -X 'GET' \
  'http://localhost:8000/stock-data-prediction?symbol=VALE3.SA&interval=1m&period=5d' \
  -H 'accept: application/json'
```
