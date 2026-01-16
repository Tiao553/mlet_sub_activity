# 🚀 Guia de Deployment

Este projeto está configurado para implantar MLflow, Airflow e a API na AWS usando Terraform e Docker.

## Pré-requisitos

- **Terraform** >= 1.0
- **AWS CLI** configurada com credenciais (`aws configure`)
- **Cliente SSH**

## 🛠️ Implantando na AWS

Fornecemos um script auxiliar para automatizar o processo de deployment.

1. **Executar o Script de Deploy**

   Vá para a raiz do projeto e execute:

   ```bash
   ./scripts/deploy_mlflow_stack.sh -var="key_name=sub-mlet-mlairflow" -auto-approve
   ```

   **O que isso faz:**
   - Inicializa o Terraform.
   - Cria recursos AWS (VPC, Security Groups, EC2, IAM Roles, S3).
   - Gera um Par de Chaves SSH (`sub-mlet-mlairflow.pem`) na raiz do projeto.
   - Sobe a stack (Airflow + MLflow + API) na instância EC2 via `user_data`.
   - Gera outputs com IPs e URLs.

2. **Acessando a Instância**

   O script exibe o **IP Público** (Elastic IP) da instância. Você também pode encontrá-lo no output do Terraform.

   ```bash
   # Conectar via SSH
   ssh -i sub-mlet-mlairflow.pem ubuntu@54.82.227.100
   ```

   *(Substitua `54.82.227.100` pelo IP real se mudar)*

   *Nota: As permissões do arquivo de chave são ajustadas automaticamente para `400`.*

## 🔍 Verificação

Após o deploy, aguarde alguns minutos para que os serviços inicializem (instalação do Docker, pull das imagens). Você pode verificar via SSH:

```bash
ssh -i sub-mlet-mlairflow.pem ubuntu@54.82.227.100 "docker ps"
```

### Acessar Serviços (Links Públicos)

- **API Gateway (Serverless)**: `https://mdrzylhavh.execute-api.us-east-1.amazonaws.com/prod/`
- **Swagger UI (Docs)**: `https://mdrzylhavh.execute-api.us-east-1.amazonaws.com/prod/docs`
- **FastAPI (EC2)**: `http://54.82.227.100:8000`
- **MLflow UI**: `http://54.82.227.100:5000`
- **Airflow UI**: `http://54.82.227.100:8080`

## ⚠️ Solução de Problemas (Troubleshooting)

**Permissões do Airflow**
Se o Airflow falhar ao iniciar com erros de "Permission denied" nos logs, o script de deploy tenta aplicar as permissões necessárias. Se precisar corrigir manualmente:

```bash
# Dentro da EC2
sudo chmod -R 777 airflow/
docker compose restart airflow-init
```

**Sincronização de DAGs (Git-Sync)**
O Airflow é configurado com um sidecar **Git-Sync**.

- DAGs são baixadas automaticamente da branch `main` do repositório GitHub.
- Intervalo de sincronização: ~30 segundos.
- Não é necessário redeployar o servidor; basta fazer push code para o GitHub.
