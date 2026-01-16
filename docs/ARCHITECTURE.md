# 🏗️ Arquitetura

Este projeto utiliza uma arquitetura nativa em nuvem na AWS para servir modelos de ML e orquestrar pipelines.

## ☁️ Infraestrutura em Nuvem (AWS)

A infraestrutura é provisionada como código (IaC) usando **HashiCorp Terraform**.

### Componentes

- **Computação (EC2)**: Uma instância `t3.medium` hospedando a stack Docker.
- **Rede (VPC)**: VPC personalizada com uma subnet pública, Internet Gateway e Route Table.
- **Segurança (SG)**:
  - `22` (SSH): Aberto para gerenciamento (Recomendado restringir ao IP do admin).
  - `5000` (MLflow): Aberto para a UI de rastreamento de ML.
  - `8080` (Airflow): Aberto para a UI de Workflow.
  - `8000` (FastAPI): Aberto para requisições de API.
- **Armazenamento (S3)**:
  - `mlet-sub-challanger-raw-data-tf`: Zona de aterrissagem (Raw) para dados buscados do Yahoo Finance.
  - `mlet-sub-challanger-delivery-data-tf`: Dados processados prontos para treinamento (Trusted/Refined).
  - `mlet-sub-challanger-mlflow-artifacts-tf`: Bucket dedicado para artefatos de modelos e rastreamento de experimentos.
- **IAM**: Perfil de Instância EC2 com acesso de leitura/escrita a esses buckets S3 específicos.

## 🐳 Stack de Aplicação (Docker)

A aplicação roda como uma stack containerizada gerenciada pelo **Docker Compose**.

### Serviços

1. **Servidor MLflow**:
   - Rastreia experimentos, métricas e modelos.
   - Banco de Dados (Backend Store): Postgres.
   - Armazenamento de Artefatos: **AWS S3** (`s3://...-mlflow-artifacts-tf`).
2. **Airflow**:
   - Orquestra pipelines de dados e retreinamento de modelos.
   - Banco de Dados (Backend Store): Postgres.
   - Componentes: Webserver, Scheduler, Init.
   - **Git-Sync**: Container "sidecar" que sincroniza DAGs/Plugins do GitHub (`https://github.com/Tiao553/mlet_sub_activity.git`) para um volume compartilhado (`git-sync-data`).
3. **PostgreSQL**:
   - Dois containers distintos para metadados do MLflow e Airflow.
4. **API (FastAPI)**:
   - Interface pública para predições.
   - Consome modelos diretamente do S3 via MLflow.

## 🔄 Fluxo de Trabalho (Workflow)

1. **Deploy**: `deploy_mlflow_stack.sh` provisiona a instância EC2.
2. **Inicialização**: O script `user_data` instala Docker, clona o repositório, corrige permissões e inicia `docker compose up`.
3. **Atualizações de DAG**: Fazer push na branch `main` do repositório atualiza automaticamente as DAGs no Airflow (atraso de aprox. 30s) via sidecar Git-Sync.
4. **Pipelines**: DAGs do Airflow executam tarefas de ML, logando métricas e modelos no MLflow.
5. **Artefatos**: Modelos são fisicamente armazenados no S3, acessíveis via UI do MLflow e API.

## 🔗 Documentação Relacionada

- [Workflow de MLOps](MLOPS_WORKFLOW.md)
- [Fluxo da API](API_FLOW.md)
- [Guia de Operações](OPERATIONS.md)
