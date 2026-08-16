# Application Run Commands

Use these commands from a normal macOS Terminal.

## Start The Dash App

```bash
cd /Users/anmolmahajan/Projects/sales-performance-ai
source .venv/bin/activate
python run_app.py
```

Open:

```text
http://127.0.0.1:8050
```

## Start Ollama For Local LLM Mode

Run this in a second Terminal window when you want to use Local LLM mode:

```bash
OLLAMA_NO_CLOUD=1 /Applications/Ollama.app/Contents/Resources/ollama serve
```

Check the installed local model:

```bash
/Applications/Ollama.app/Contents/Resources/ollama list
```

Expected model:

```text
qwen3:4b-instruct
```

## Run Both Services

Terminal 1:

```bash
cd /Users/anmolmahajan/Projects/sales-performance-ai
source .venv/bin/activate
python run_app.py
```

Terminal 2:

```bash
OLLAMA_NO_CLOUD=1 /Applications/Ollama.app/Contents/Resources/ollama serve
```

## Health Checks

Dash app:

```bash
curl http://127.0.0.1:8050/healthz
curl http://127.0.0.1:8050/readyz
```

Ollama:

```bash
curl http://127.0.0.1:11434/api/tags
```

## Stop The Services

Press `CTRL+C` in each Terminal window running a service.

## Optional Container Run

```bash
cd /Users/anmolmahajan/Projects/sales-performance-ai
docker compose up --build
```

Open:

```text
http://127.0.0.1:8050
```
