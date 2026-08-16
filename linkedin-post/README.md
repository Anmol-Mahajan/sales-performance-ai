# LinkedIn post package

This folder contains a ready-to-edit LinkedIn post, five recommended screenshots, and one optional salesperson-detail screenshot captured from the local Dash application.

## Files

- `linkedin-post.md`: proposed post text
- `image-captions.md`: short captions and accessible alt text
- `screenshots/00-sales-manager-overview.png`: sales and customer performance overview
- `screenshots/01-local-llm-question.png`: workbook-relative meeting query verified through the local question workflow
- `screenshots/02-revenue-model-purpose.png`: prediction purpose and limitations
- `screenshots/03-model-validation.png`: candidate model comparison
- `screenshots/04-feature-impact.png`: feature importance and feature-set experiment
- `screenshots/05-sales-person-overview.png`: optional salesperson performance, operational health and recommendations view
- `capture_screenshots.py`: offline, localhost-only capture script

## Suggested image order

Use images `00` to `04` in numbered order. They move from the wider manager experience to the local question workflow, model boundary, validation results and the effect of adding new features. Image `05` is an optional supporting view when more detail on the whole salesperson and customer relationship is useful.

The metrics in the post describe the workbook and model state at the time these screenshots were captured. They are demonstration results, not production evidence.

## Local-only boundary

The capture script only connects to the Dash app at `127.0.0.1:8050` and a local Chrome debugging session at `127.0.0.1:9223`. It does not use the internet or an external API. The Qwen model is run locally through Ollama, and the revenue model is trained and saved locally.

## Re-capturing after model changes

Start the Dash app:

```bash
cd /Users/anmolmahajan/Projects/sales-performance-ai
source .venv/bin/activate
python run_app.py
```

In a second terminal, start the installed Chrome in local headless mode:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new \
  --disable-gpu \
  --disable-background-networking \
  --disable-component-update \
  --disable-default-apps \
  --disable-sync \
  --disable-features=OptimizationHints,MediaRouter,Translate,SyncService \
  --host-resolver-rules="MAP * 0.0.0.0, EXCLUDE localhost" \
  --metrics-recording-only \
  --safebrowsing-disable-auto-update \
  --user-data-dir=/private/tmp/sales-ai-linkedin-chrome \
  --remote-debugging-port=9223 \
  --window-size=1440,1100 \
  about:blank
```

Then run:

```bash
cd /Users/anmolmahajan/Projects/sales-performance-ai
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python linkedin-post/capture_screenshots.py
```
