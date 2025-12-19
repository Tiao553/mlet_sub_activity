# Tech Challenger 4 - Deep Learning

[link do video de apresentacao](https://youtu.be/v03U9tBDizg)

> Projeto para previsão de preços de ações com base em séries temporais históricas, utilizando uma rede neural recorrente (LSTM) implementada em TensorFlow/Keras. A solução inclui uma API RESTful para inferência e infraestrutura como código para deploy em ambiente cloud AWS.

---

## Sumário

- [Descrição](#descrição)
- [Objetivo](#objetivo)
- [Arquitetura do Projeto](#arquitetura-do-projeto)
- [Requisitos](#requisitos)
- [Instalação Local](#instalação-local)
- [Uso da API](#uso-da-api)
- [Exemplo de Request e Response](#exemplo-de-request-e-response)
- [Treinamento e Reutilização do Modelo](#treinamento-e-reutilização-do-modelo)
- [Infraestrutura como Código (IaC)](#infraestrutura-como-código-iac)
- [Deploy com AWS Fargate](#deploy-com-aws-fargate)
- [To-Do e Melhorias Futuras](#to-do-e-melhorias-futuras)
- [Licença](#licença)

---

## 📘 Descrição

Este projeto propõe uma solução de aprendizado de máquina para previsão do próximo valor diário de uma ação (exemplo: TSLA), utilizando dados históricos de fechamento. O modelo de predição é uma rede neural do tipo **LSTM** (Long Short-Term Memory), especializada em dados sequenciais e séries temporais.

A aplicação oferece:

- Uma **API RESTful** que recebe os dados históricos e retorna a previsão do próximo dia.
- Um pipeline de pré-processamento e coleta de dados históricos.
- Uma estrutura modularizada em Python.
- Infraestrutura como código para deployment automatizado via AWS.

---

## 🎯 Objetivo

Criar uma solução capaz de:

- Carregar dados históricos de ações.
- Processar e alimentar uma LSTM treinada para prever o valor do dia seguinte.
- Disponibilizar a predição via API.
- Implantar a aplicação em ambiente escalável na AWS (Fargate, Lambda, API Gateway).

---

## 🧱 Arquitetura do Projeto

```sh
.
├── api                          # Código da aplicação
│   ├── app                     # Lógica de aplicação
│   │   ├── api.py              # Endpoints da API
│   │   ├── config/logger.py    # Configuração de logs
│   │   ├── services/           # Serviços para predição e fetch
│   │   └── model/              # Modelos .h5 salvos
│   ├── dockerfile              # Containerização da aplicação
│   ├── main.py                 # Entry point (Flask app ou FastAPI)
│   └── requirements.txt        # Dependências Python
├── infrastructure              # Código Terraform para AWS
├── model                       # Modelos adicionais ou versões
├── notebooks                   # Análises e experimentos
└── readme.md                   # Este arquivo
```

---

## 🔧 Requisitos

- Python 3.10+
- Docker
- Terraform >= 1.4
- AWS CLI configurado
- TensorFlow >= 2.x
- Ambiente virtual recomendado

---

## 🚀 Instalação Local

```bash
# Clonar o repositório
git clone https://github.com/seu-usuario/tech-challenger-4.git
cd tech-challenger-4

# Ativar ambiente virtual
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows

# Instalar dependências
pip install -r api/requirements.txt

# Rodar a API
```

## 🐳 Execução com Docker

Alternativamente, você pode rodar a aplicação utilizando Docker:

1. **Construir a imagem:**

```bash
docker build -t tech-challenge-api -f api/dockerfile ./api
```

2. **Rodar o container:**

```bash
docker run -d -p 80:80 --name tech-challenge-app tech-challenge-api
```

3. **Verificar logs (opcional):**

```bash
docker logs -f tech-challenge-app
```

## 📡 Uso da API

A API possui o endpoint principal `/stock-data-prediction` (GET) que aceita os seguintes parâmetros:

- `symbol` (obrigatório): Código da ação (ex: `VALE3.SA`, `TSLA`).
- `start_date` (opcional): Data de início (YYYY-MM-DD).
- `end_date` (opcional): Data de fim (YYYY-MM-DD).
- `interval` (opcional): Intervalo dos dados (padrão: `1m`).
- `period` (opcional): Período de dados para busca (ex: `7d`).
- `auto_adjust` (opcional): Ajuste automático de preços (default `True`).

### Exemplo de Request e Response

**Request (cURL):**

```bash
curl -X 'GET' \
  'http://localhost:80/stock-data-prediction?symbol=VALE3.SA&interval=1m&period=5d' \
  -H 'accept: application/json'
```

**Response (JSON):**

```json
22.45
```

*(O retorno é um valor float representando o preço previsto)*
