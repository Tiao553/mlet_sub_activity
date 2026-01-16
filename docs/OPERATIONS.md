# ⚙️ Manual de Operações

Este documento fornece instruções para gerenciar a infraestrutura do projeto e scripts operacionais localizados em `scripts/` e `infrastructure/`.

## 🖥️ Gerenciamento da Instância EC2

Utilize o script `scripts/manage_ec2.py` para controlar o servidor MLflow/Airflow. Isso ajuda a reduzir custos pausando a instância quando não estiver em uso.

**Uso**:

```bash
# Iniciar a instância
python3 scripts/manage_ec2.py start

# Parar (Pausar) a instância
python3 scripts/manage_ec2.py stop

# Verificar Status
python3 scripts/manage_ec2.py status
```

*Nota: O script detecta automaticamente a instância baseada nas tags (`Project=sub-challanger` e `Name=*mlflow-airflow-server`).*

## 🚀 Deployment & Manutenção

### Implantando a Stack (Deploy)

Para provisionar a infraestrutura do zero na AWS:

```bash
./scripts/deploy_mlflow_stack.sh
```

### Destruindo a Stack

Para derrubar todos os recursos (CUIDADO: buckets S3 específicos podem ser mantidos se não estiverem vazios):

```bash
./scripts/destroy_mlflow_stack.sh
```

### Acesso SSH

Gere um novo par de chaves SSH se necessário (as chaves são salvas localmente):

```bash
./scripts/generate_pem.sh
```

### Sincronização Git (Git Sync)

O `git-sync` roda como um sidecar no Docker. Para testar a lógica ou forçar uma sincronização manual de código se estiver rodando scripts locais:

```bash
./scripts/git_sync.sh
```

## 🧹 Limpeza (Cleanup)

Para forçar a limpeza de recursos docker ou arquivos temporários na máquina local:

```bash
python3 scripts/force_cleanup.py
```
