# Workflow de MLOps & Arquitetura

## 1. Estratégia de Roteamento Dinâmico da API

Para lidar com N modelos (um para cada ação, período e intervalo), a API constrói dinamicamente a URI do modelo específico em tempo de execução.

### Convenção de Nomenclatura

`model_{SÍMBOLO}_{PERÍODO}_{INTERVALO}` (ex: `model_PETR4_1y_1d`)

### Fluxo de Requisição

1. **Requisição do Usuário**: `GET /predict?symbol=PETR4&period=1y&interval=1d`
2. **Carregamento do Modelo**: A API verifica no MLflow se existe um modelo com o nome correspondente e a tag `@champion` (Produção).
3. **Inferência**: Retorna a predição usando o modelo campeão específico carregado.

## 2. Workflow de Promoção (HMG -> Produção)

### Passo A: Homologação Automatizada (Airflow)

* **Gatilho**: DAG Diária/Semanal (`model_promotion_dag`) no Apache Airflow.
* **Lógica**:
    1. Escaneia todos os Experimentos do MLflow.
    2. Identifica a execução (run) com as **Melhores Métricas** (menor RMSE e MAE).
    3. Registra essa versão do modelo no Model Registry.
    4. Atribui o alias **`@HMG`** (Homologação).
* **Resultado**: Modelos validados aguardam na área de "Homologação".
* **Script Fonte**: `models/management/promote_model.py`.

### Passo B: Eleição para Produção (Gate Humano)

* **Gatilho**: Operador Humano / Cientista de Dados.
* **Lógica**:
    1. Revisa os modelos vinculados ao alias `@HMG` na UI do MLflow.
    2. Seleciona a versão específica para entrar em Produção (Live).
    3. Executa o script de aprovação.
    4. O script atribui o alias **`@champion`** ao modelo escolhido.
* **Script Fonte**: `models/management/approve_model.py`.

## 3. Mecanismo de Rollback

* **Script Fonte**: `models/management/rollback_model.py`.
* **Ação**: Reatribui o alias `@champion` para uma versão anterior conhecida como estável ("last good known"), permitindo recuperação rápida em caso de degradação de performance.

## 🔗 Guias Detalhados

* Para saber como executar estes scripts, veja **[Experimentos (EXPERIMENTS.md)](EXPERIMENTS.md)**.
* Para comandos operacionais, veja **[Operações (OPERATIONS.md)](OPERATIONS.md)**.
