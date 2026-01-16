# Atividade Substitutiva da Pós-Graduação MLET03 da FIAP

[link do video de apresentacao](https://youtu.be/8pmkMUNIThI)

> **Projeto de Operacionalização de Modelo de Previsão de Ações (Stock Prediction MLOps)**
>
> Este projeto é uma atividade substitutiva para a disciplina MLET03. O objetivo é operacionalizar um modelo de Deep Learning (LSTM) para previsão de preços de ações, integrando-o com **MLflow** para rastreamento de experimentos e gerenciamento de modelos, além de preparar a infraestrutura para deploy e serving na AWS.

---

## Sumário

- [Descrição](#descrição)
- [Objetivo](#objetivo)
- [Acesso Público (AWS)](#acesso-público-aws)
- [Documentação Completa](#documentação-completa)
- [Arquitetura do Projeto](#arquitetura-do-projeto)
- [Instalação e Execução](#instalação-e-execução)

---

## 📘 Descrição

Este projeto utiliza redes neurais (LSTM, GRU, BiLSTM) para prever o valor de fechamento de ações (ex: PETR4.SA, VALE3.SA) no futuro. O diferencial é a **operacionalização MLOps completa**, focando em:

1. **Rastreabilidade**: Uso do MLflow para registrar métricas, parâmetros e artefatos de treinamento no S3.
2. **Reprodutibilidade**: Estruturação dos experimentos (Grid Search) para cobrir diversos cenários.
3. **Model Serving**: API capaz de carregar dinamicamente o melhor modelo (Campeão) para cada par Ação/Intervalo.

---

## 🎯 Objetivo

Criar uma solução robusta que não apenas treine um modelo, mas gerencie seu ciclo de vida (Treino -> Homologação -> Produção):

- **Infraestrutura**: Containerização com Docker e orquestração dos serviços (API + MLflow + Airflow) na AWS.
- **Automação**: Pipelines para promoção automática de modelos e deploy de infraestrutura via Terraform.

---

## 🌐 Acesso Público (AWS)

O projeto está implantado e acessível publicamente:

- **API (Gateway Serverless)**: [https://mdrzylhavh.execute-api.us-east-1.amazonaws.com/prod/](https://mdrzylhavh.execute-api.us-east-1.amazonaws.com/prod/)
- **Documentação da API (Swagger/Redoc)**: [https://mdrzylhavh.execute-api.us-east-1.amazonaws.com/prod//docs](https://mdrzylhavh.execute-api.us-east-1.amazonaws.com/prod//docs)
- **MLflow UI**: [http://54.82.227.100:5000](http://54.82.227.100:5000)
- **Airflow UI**: [http://54.82.227.100:8080](http://54.82.227.100:8080)

---

## 📚 Documentação Completa

Reescrevemos toda a documentação para facilitar o entendimento e operação:

### 🏗️ Arquitetura & Infra

* **[Arquitetura do Sistema](docs/ARCHITECTURE.md)**: Visão de containers, networking AWS e fluxo de dados.
- **[Guia de Operações (Ops)](docs/OPERATIONS.md)**: Manual para deploy, gerenciamento de EC2 e scripts de manutenção.
- **[Guia de Deployment](docs/DEPLOYMENT.md)**: Passo a passo para subir a stack do zero na AWS via Terraform.

### 🧪 MLOps & Desenvolvimento

* **[Experimentos & Treinamento](docs/EXPERIMENTS.md)**: Como executar Grid Search e treinar novos modelos.
- **[Workflow de MLOps](docs/MLOPS_WORKFLOW.md)**: Entenda o ciclo de vida dos modelos (Dev -> HMG -> Prod) e a estratégia de promoção.
- **[Decisões Técnicas](docs/technical_decisions.md)**: Justificativa para escolha de modelos, parâmetros e design.

### 🔌 API

* **[Fluxo de Uso da API](docs/API_FLOW.md)**: Diagrama visual de como a API processa requisições.
- **[Referência da API](docs/API_REFERENCE.md)**: Detalhes dos endpoints, parâmetros e exemplos de resposta.

---

## 🧱 Arquitetura do Projeto

```sh
.
├── api                          # Código da aplicação (FastAPI)
│   ├── app                     # Lógica de aplicação
│   │   ├── api.py              # Endpoints da API
│   │   ├── services/           # Serviços para predição e fetch
│   │   └── model/              # (Legado) Modelos locais
│   ├── dockerfile              # Dockerfile da API
├── infrastructure              # Infraestrutura (Terraform/Docker)
├── models                      # Scripts de treinamento e Gestão MLflow
│   ├── execution/              # Scripts de execução (Grid Search)
│   ├── training/               # Scripts de treinamento (Train Model)
│   └── management/             # Scripts de promoção/aprovação
├── notebooks                   # Análises exploratórias
├── docker-compose.yml          # Orquestração (API + MLflow + Airflow)
└── readme.md                   # Este arquivo
```

---

## 🚀 Instalação e Execução

### Usando Docker Compose (Local)

O ambiente inclui a API, Airflow e o servidor MLflow.

```bash
# Subir os serviços
docker-compose up -d --build
```

- **API**: <http://localhost:8000>
- **MLflow UI**: <http://localhost:5000>
- **Airflow UI**: <http://localhost:8080>

### Deployment na AWS

Consulte o **[Guia de Deployment](docs/DEPLOYMENT.md)** para instruções detalhadas sobre como provisionar a infraestrutura usando o script `./scripts/deploy_mlflow_stack.sh`.
