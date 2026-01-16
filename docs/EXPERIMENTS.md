# 🧪 Experimentos & Treinamento de Modelos

Este documento detalha o fluxo de trabalho para treinar modelos, executar buscas em grade (grid searches) e promover modelos através do pipeline de MLOps.

## 🔎 Grid Search (Otimização de Hiperparâmetros)

O script `models/execution/run_grid_search.py` é o ponto de entrada para o treinamento massivo de modelos. Ele explora um espaço de hiperparâmetros definido através de múltiplos cenários de negócio.

**Uso**:

```bash
python3 models/execution/run_grid_search.py
```

### Espaço de Busca

A busca em grade amostra aleatoriamente parâmetros de:

* **Frameworks**: TensorFlow (Keras), PyTorch
* **Arquiteturas**: LSTM, Bi-LSTM, GRU
* **Hiperparâmetros**: Tamanho da Sequência (Sequence Length), Tamanho do Lote (Batch Size), Épocas (Epochs), Taxa de Aprendizado (Learning Rate), Unidades Ocultas (Hidden Units), Dropout, Camadas.

### Cenários de Negócio

Combinações de:

* **Símbolos**: `VALE3.SA`, `AAPL`, `NVDA`, `ITSA4.SA`, `WEGE3.SA`, `^GSPC`
* **Períodos (Histórico)**: 1mo (1 mês) até o máximo disponível
* **Intervalos (Granularidade)**: 5m, 15m, 30m, 60m, 1d (diário), 1wk (semanal), 1mo (mensal)

*Nota: Combinações inválidas (ex: intervalo de 1m com período de 10y) são automaticamente ignoradas pela API do Yahoo Finance e tratadas pelo script.*

## 🏋️ Requisições de Treinamento

O treinamento direto de um modelo específico pode ser feito via `models/training/train_model_grid.py`. Este script é geralmente invocado pelo orquestrador da grid search, mas pode ser executado isoladamente para testes.

```bash
python3 models/training/train_model_grid.py --symbol PETR4.SA --period 1y --interval 1d ...
```

## 🏆 Promoção de Modelos

Os modelos são gerenciados através de um sistema de "alias" (apelidos) no MLflow (`@HMG`, `@champion`).

### Scripts de Gestão (`models/management/`)

1. **Promover para Homologação**:
    Analisa os experimentos e busca o melhor modelo (menor RMSE/MAE) para cada cenário, marcando-o com a tag `@HMG`.

    ```bash
    python3 models/management/promote_model.py
    ```

2. **Aprovar para Produção**:
    Promove um modelo marcado como `@HMG` para `@champion`, tornando-o o modelo ativo na API.

    ```bash
    python3 models/management/approve_model.py
    ```

3. **Rollback (Reversão)**:
    Reverte o alias `@champion` para uma versão anterior funcional caso surjam problemas.

    ```bash
    python3 models/management/rollback_model.py
    ```

4. **Relatórios**:
    Gera um relatório em markdown sobre a performance atual dos modelos.

    ```bash
    python3 models/management/generate_report.py
    ```
