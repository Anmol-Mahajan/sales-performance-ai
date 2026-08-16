# Future Azure Container Apps Deployment

This directory makes the application ready for a future private Azure Container Apps deployment. It does not deploy anything and introduces no cloud API into the application.

## Important Boundary Change

The current application is local-only. Deploying it to Azure means the workbook, questions, model artefacts and application logs are processed outside the Mac. That is a material privacy-boundary change and must be approved before any real organisation data is uploaded.

Before deployment, agree and document:

- data owner and lawful purpose
- approved Azure tenant, subscription and region
- private network and user authentication design
- workbook and model storage encryption
- managed identity and least-privilege access
- log redaction, retention and deletion
- backup, recovery and model-retention policy
- customer and employee data residency requirements
- vulnerability scanning and dependency patching

## Image Boundary

`.dockerignore` excludes Excel workbooks and trained model files. The image therefore contains application code and configuration only. Data and model artefacts must be mounted at runtime.

The supplied Container Apps template expects two Azure Files mounts:

- `/mnt/sales-data`: read-only workbook storage
- `/mnt/models`: writable model artefact storage

Configure both storage entries on the Container Apps managed environment, then replace every `<placeholder>` in `azure-container-app.yaml`.

## Runtime Design

- Gunicorn binds to container port `8050`.
- One worker is used by default to avoid loading the workbook and models multiple times.
- One replica is used because the current app trains and saves models during startup. Separate training from serving before enabling horizontal scaling.
- Four threads allow concurrent Dash requests within that process.
- `/healthz` reports process health.
- `/readyz` confirms that the workbook and metric configuration are mounted.
- Ingress is internal by default in the template.
- The local Ollama model is not included in the image.

Workbook mode remains available without Ollama. Running Qwen in Azure would require a separately approved private inference design and sufficient memory; do not silently replace it with a hosted model or external API.

## Local Container Check

With Docker installed, the same image can be tested locally:

```bash
docker compose up --build
```

The compose file mounts `./data` read-only and `./models` as writable storage. Open `http://127.0.0.1:8050`, then check:

```text
http://127.0.0.1:8050/healthz
http://127.0.0.1:8050/readyz
```
