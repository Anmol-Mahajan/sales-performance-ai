# Sales Performance AI

Local-only Plotly Dash application for MSP sales performance, explainable manager insights, customer billing, contract renewal health, and local model comparison.

The Sales Manager Portal includes an offline **Ask your sales data** interface with a deterministic workbook engine and an optional local LLM mode.

## Local-Only Model Boundary

This project must only use local files inside the project folder.

Allowed:
- Local Excel files
- Local Python scripts
- Local pandas processing
- Local scikit-learn models
- An optional local Ollama model after a one-time explicit download
- Local model artifacts saved under `/models`
- Local Plotly Dash app running on localhost

Not allowed:
- Internet data
- Web scraping
- External APIs
- OpenAI API calls
- Cloud ML services
- Hosted model inference
- Sending customer, contract, billing, salesperson, or notes data outside the machine

The application does not use the internet during normal operation. Installing Ollama and downloading a model are explicit, one-time setup actions. The app does not call external APIs or hosted inference. Revenue models are trained only on the local Excel workbook and saved locally. The optional LLM runs locally and is grounded with workbook results; it is not fine-tuned on customer data. No customer or salesperson data leaves the machine.

## Project Structure

```text
sales-performance-ai/
├── .dockerignore
├── .env.example
├── .gitignore
├── data/
│   ├── MSP_Sales_Performance_Raw_Data_With_Common_Metrics.xlsx
│   └── MSP_Sales_Performance_Demo_Data_Updated_With_Contracts.xlsx
├── config/
│   ├── sales_metrics.yaml
│   └── query_intents.yaml
├── docs/
│   └── skills_updated_with_contracts.md
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── validation.py
│   ├── features.py
│   ├── model_training.py
│   ├── model_utils.py
│   ├── query_planning.py
│   ├── query_execution.py
│   ├── settings.py
│   ├── time_periods.py
│   ├── local_llm.py
│   └── pipeline_forecasting.py
├── app/
│   ├── __init__.py
│   ├── app.py
│   ├── manager_portal.py
│   ├── data_scientist_portal.py
│   └── assets/
│       └── styles.css
├── models/
├── scripts/
│   ├── populate_operational_data.py
│   └── setup_local_llm.sh
├── tests/
│   ├── test_query_planning.py
│   ├── test_question_answering.py
│   └── test_runtime.py
├── linkedin-post/
├── deploy/
│   ├── README.md
│   └── azure-container-app.yaml
├── Dockerfile
├── compose.yaml
├── requirements.txt
├── README.md
└── run_app.py
```

## Setup

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/<your-github-username>/sales-performance-ai.git
cd sales-performance-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you already have the project locally, start from the project folder instead:

```bash
cd /Users/anmolmahajan/Projects/sales-performance-ai
source .venv/bin/activate
```

The required packages are local Python libraries: pandas, numpy, scikit-learn, Plotly Dash, dash-bootstrap-components, openpyxl, joblib, PyYAML, and Gunicorn for container serving.

Optional local model libraries are supported if already installed in your environment:
- xgboost
- lightgbm
- catboost

The app does not require those optional libraries. If they are missing, model training continues with scikit-learn models.

## Ask Questions Locally

The question interface has two local modes:

- **Workbook engine:** fast deterministic pandas calculations with source tables.
- **Local LLM:** Qwen interprets freer phrasing and explains a compact, relevant slice of workbook results. Numeric results remain grounded in pandas calculations.

Example questions:

- `Who are the top performers?`
- `How is Alice performing?`
- `Which contracts need health checks?`
- `Show renewals within 60 days`
- `Where are the best customer whitespace opportunities?`
- `Which referral partnerships convert best?`
- `Who sat the most meetings last week?`
- `Which opportunity notes are waiting for a response?`
- `Show critical findings from customer meetings`
- `What stage is the project linked to OPP0001?`
- `Which opportunities have blocked projects or tasks?`
- `How achievable is the pipeline revenue target?`
- `What actions can help cover the pipeline gap?`

The app shows the local source for each answer and does not send the question, context, or answer outside the machine.

### Structured Intent Routing

Questions are first converted into a local `QueryPlan`. The plan extracts independent parts rather than selecting one broad keyword branch:

- domain, such as opportunities, meetings, projects, contracts, customers, or performance
- operation, such as list, rank, status, forecast, or explain
- salesperson and customer identifiers validated against workbook records
- filters such as opportunity type and status
- time scope such as upcoming, current year, or last complete week
- result limit and sort order

For example, `What are the upcoming cross-sell opportunities for Chloe Singh?` becomes an opportunity-list plan filtered to Chloe Singh, `OpportunityType = Cross-sell`, `Stage = Open`, no close date, and an expected close date on or after the workbook snapshot. Pandas executes the validated plan. The answer displays an **Interpreted as** strip so the manager can inspect those filters.

Synonyms are maintained in `config/query_intents.yaml`. Adding wording such as a new opportunity-type alias does not require another question-specific Python branch. If wording has two valid meanings, such as recorded cross-sell pipeline versus customer whitespace, the app asks for clarification instead of guessing.

Project questions are resolved from the `Projects`, `OpportunityTickets`, and `TicketTasks` sheets before any optional language-model step. Direct project, ticket, and task lookups remain deterministic in both question modes so a local LLM cannot rewrite verified status. Scoped action questions use the recorded overdue dates, blockers, delivery health, ticket counts, and task counts to produce traceable suggestions.

When a question is ambiguous, causal, or missing its subject, scope, period, comparison, or objective, the app does not guess. It asks the manager to clarify the request and displays examples of the identifiers or details needed. This is particularly important because the current chat is single-turn and does not retain earlier conversational references such as “this project” or “tell me more”.

## Optional Local LLM Setup

The recommended model for this 16 GB Apple Silicon Mac is `qwen3:4b-instruct`. It is approximately 2.5 GB and runs through Ollama on localhost.

Install the official Ollama macOS application once, then run:

```bash
cd /Users/anmolmahajan/Projects/sales-performance-ai
./scripts/setup_local_llm.sh
```

Start Ollama with cloud features disabled:

```bash
OLLAMA_NO_CLOUD=1 /Applications/Ollama.app/Contents/Resources/ollama serve
```

The Dash app invokes the local `ollama` executable directly. It does not use the Ollama Python client or a hosted API. To choose another already-downloaded local model:

```bash
export SALES_AI_LOCAL_MODEL=qwen3:4b-instruct
python run_app.py
```

The local LLM is used for question interpretation and explanation, not conventional model training. Fine-tuning on this small workbook would be statistically weak and risks memorising names and rows. The scikit-learn revenue model remains the organisation-specific model trained from the workbook.

## Operational Sales Data

The primary local workbook contains synthetic operational demonstration data in these linked sheets:

- `Meetings`: salesperson notes, customer context, opportunity links, follow-ups, and critical findings.
- `OpportunityNotes`: customer and internal notes, response status, waiting age, and escalation severity.
- `Projects`: project stage, delivery health, completion, milestones, and blockers linked to opportunities.
- `OpportunityTickets`: ticket ownership, status, priority, due dates, and escalation flags.
- `TicketTasks`: task-level progress and blockers linked to tickets and opportunities.

Meeting rows in the Sales Manager Portal can be opened to inspect salesperson notes and highlighted critical findings. Opportunity rows open their related notes, project, tickets, and task statuses.

The demonstration records can be regenerated deterministically without internet access:

```bash
python scripts/populate_operational_data.py
```

This replaces only the generated operational and reference sheets in the local workbook. It does not call an API or use real customer data.

The generator also maintains a marked synthetic current-year layer:

- `MonthlyPerformance` contains 2026 year-to-date salesperson results through the local snapshot date.
- `Opportunities` contains 2026 opportunities with pipeline stage, expected close date, probability, forecast category, stage age, next step, and risk.

## Pipeline Revenue Forecast

The manager portal compares 2026 YTD recognised revenue and probability-adjusted open pipeline with annual targets. Forecast revenue is not the raw pipeline total. Each current opportunity is adjusted using:

- recorded sales-stage probability
- historical salesperson conversion
- the local opportunity classifier only when its holdout ROC-AUC clears the quality guardrail
- pipeline risk and time in stage
- unanswered customer notes
- critical meeting and note findings

The app reports open pipeline, weighted pipeline, forecast year-end revenue, forecast gap, coverage, and an achievability score. It also creates evidence-based actions for pipeline creation, customer responses, stalled opportunities, and priority next steps. These are local scenarios rather than guaranteed outcomes.

The pipeline classifier is saved locally to:

```text
models/best_pipeline_model.joblib
```

## Run The App

For a short command-only reference, see `RUN_COMMANDS.md`.

```bash
cd sales-performance-ai
source .venv/bin/activate
python run_app.py
```

Open:

```text
http://127.0.0.1:8050
```

The Dash server binds to `127.0.0.1`, so it runs on localhost.

## Test

```bash
cd sales-performance-ai
source .venv/bin/activate
python -m pytest
```

If `pytest` is not installed in the environment, install it locally:

```bash
pip install pytest
python -m pytest
```

## Publish To GitHub

Create a repository named `sales-performance-ai`, commit the project, and push it:

```bash
cd sales-performance-ai
git init
git branch -M main
git add .
git commit -m "Initial sales performance AI project"
git remote add origin https://github.com/<your-github-username>/sales-performance-ai.git
git push -u origin main
```

With the GitHub CLI installed and authenticated, the repository can be created and pushed in one flow:

```bash
cd sales-performance-ai
gh repo create sales-performance-ai --private --source=. --remote=origin --push
```

Use `--public` instead of `--private` only after confirming the included workbook and screenshots are safe to publish.

## Runtime Configuration

Local defaults require no environment variables. The following settings allow the same code to run in a private container:

| Variable | Local default | Purpose |
|---|---|---|
| `SALES_AI_RUNTIME_MODE` | `local` | Selects accurate local or private-container boundary wording |
| `HOST` | `127.0.0.1` | Dash development-server bind address |
| `PORT` | `8050` | HTTP port |
| `SALES_AI_WORKBOOK_PATH` | Primary workbook under `data/` | Runtime workbook mount |
| `SALES_AI_METRICS_PATH` | `config/sales_metrics.yaml` | Metric semantics |
| `SALES_AI_QUERY_INTENTS_PATH` | `config/query_intents.yaml` | Intent and synonym vocabulary |
| `SALES_AI_MODEL_DIR` | `models/` | Writable local model artefacts |

`/healthz` reports process health. `/readyz` confirms that the configured workbook and metrics file are available.

## Local Container Check

The container image does not include Excel workbooks or trained model artefacts. They are excluded by `.dockerignore` and mounted at runtime:

```bash
docker compose up --build
```

Open `http://127.0.0.1:8050`. The compose file mounts `./data` read-only and `./models` as writable storage. Gunicorn uses one worker by default so the workbook and models are not loaded into several processes on a small machine.

## Future Azure Container Apps Boundary

`deploy/azure-container-app.yaml` is a private-ingress template for a future Azure Container Apps deployment. It expects separate Azure Files mounts for the workbook and model artefacts and uses managed identity for the container registry.

Deploying to Azure changes the local-only boundary: workbook data, questions, model files and logs would be processed outside this Mac. Do not upload real organisation data until the data owner has approved the tenant, region, identity, private networking, authentication, encryption, logging, retention and residency controls. See `deploy/README.md` for the deployment checklist.

The container does not include Ollama or Qwen. Workbook mode works without them. Any future private LLM container requires a separately approved design and must not be silently replaced with a hosted model or external API.

## Train The Model Locally

The Model Analysis tab trains revenue prediction models during app startup using lagged monthly features from the local workbook. Current-month revenue is the target; predictors come from prior months and static salesperson attributes. Lagged meeting-purpose and opportunity-note counts are included where historically available. Current project or ticket status is not used to predict historical revenue. The final months are held out chronologically, avoiding same-period target leakage and random future-to-past mixing.

The accuracy experiment tracker compares the original lagged workbook feature set with the operational meeting and note features on the same chronological holdout. It reports whether RMSE improved or worsened; operational features are not assumed to help merely because they are available.

The saved revenue model is selected across both feature sets using the lowest chronological-holdout RMSE. If operational meeting or note features worsen later-period accuracy, they are excluded from the saved model. The Model Analysis tab explains the prediction target, lag timing, temporal split, baseline comparison, feature-set decision, feature importance, and current evidence limitations. Feature importance describes predictive reliance and association, not causation.

The local models compared are:
- DummyRegressor baseline
- LinearRegression
- RandomForestRegressor
- GradientBoostingRegressor
- XGBoost, LightGBM, and CatBoost only if already installed

Metrics shown:
- MAE
- RMSE
- R²

The best model is saved locally to:

```text
models/best_revenue_model.joblib
```

You can also train from Python:

```bash
python - <<'PY'
from src.data_loader import load_sales_data
from src.model_training import train_revenue_models

data = load_sales_data()
result = train_revenue_models(data)
print(result.model_comparison)
print(result.model_path)
PY
```

## Confirm It Is Local-Only

Check the code paths:
- `src/data_loader.py` reads only the Excel workbook under `data/`.
- `src/metric_config.py` safely loads local YAML semantics and recomputes supported derived metrics.
- `src/model_training.py` trains local tabular models only.
- `src/model_utils.py` saves model artifacts under `models/`.
- `src/local_llm.py` invokes only the local Ollama executable and passes bounded workbook context through local process input.
- No module imports OpenAI SDKs, cloud SDKs, HTTP clients, scraping tools, or hosted inference clients.

The current source workbook and metric semantics were copied from:

```text
/Users/anmolmahajan/Projects/MSP_Sales_Performance_Raw_Data_With_Common_Metrics.xlsx
/Users/anmolmahajan/Projects/sales_metrics.yaml
```

The implementation notes were copied from:

```text
/Users/anmolmahajan/Projects/skills.md
```
