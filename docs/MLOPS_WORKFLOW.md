# MLOps Workflow & Architecture

## 1. Dynamic API Routing Strategy

To handle N models (one per stock, period, and interval), the API dynamically constructs the specific model URI at runtime.

### Naming Convention

`model_{SYMBOL}_{PERIOD}_{INTERVAL}` (e.g., `model_PETR4_1y_1d`)

### Request Flow

1. **User Request**: `GET /predict?symbol=PETR4&period=1y&interval=1d`
2. **Model Loading**: API checks for models tagged as `@champion` (Production).
3. **Inference**: Returns prediction using the specific champion model.

## 2. Promotion Workflow (HMG -> Production)

### Step A: Automated Homologation (Airflow)

* **Trigger**: Daily/Weekly DAG (`model_promotion_dag`).
* **Logic**:
    1. Scans MLflow Experiments.
    2. Identifies run with **Best Metrics** (RMSE & MAE).
    3. Registers the model version.
    4. Tags it with alias **`@HMG`** (Homologation).
* **Result**: Valid models are waiting in the "HMG" holding area.
* **Script**: `promote_to_hmg.py`.

### Step B: Production Election (Human Gate)

* **Trigger**: Human Operator / Data Scientist.
* **Logic**:
    1. Review models tied to the `@HMG` alias in MLflow UI.
    2. Select the specific version to go Live.
    3. Run approval script.
    4. Script assigns the **`@champion`** alias.
* **Script**: `approve_model.py`.

## 3. Rollback Mechanism

* **Script**: `rollback_model.py`.
* **Action**: Re-assigns `@champion` to a previous known good version.
