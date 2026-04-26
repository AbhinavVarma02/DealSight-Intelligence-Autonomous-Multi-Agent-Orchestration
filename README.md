# DealSight Intelligence

**Autonomous Multi-Agent Deal Hunting**

DealSight Intelligence is an end-to-end AI deal-finder. It scans live deal feeds, prices each candidate product through an ensemble of pricing agents (a frontier LLM with retrieval, a fine-tuned specialist model, and a local deep neural network), keeps a memory of opportunities, and can send alerts when a deal beats its estimated value by a configurable margin.

The system is built so that every external dependency is optional. If Modal, Pushover, the vector store, or trained model artifacts are not available, the agents fall back to safe behavior instead of crashing.

## Highlights

- **Multi-agent orchestration** — Scanner, Frontier, Specialist, Neural Network, Ensemble, Planning, and Messaging agents
- **Retrieval-augmented pricing** — Chroma vector store of historical priced products supplies similar-item context to the frontier model
- **Fine-tuned specialist** — QLoRA-tuned Llama adapter served on Modal for second-opinion pricing
- **Local deep neural network** — residual MLP over hashed text features for a fully-offline price estimate
- **Dry-run by default** — no real alerts go out until you explicitly opt in
- **Tested core** — unit tests cover deals, scanner, planner, datasets, and the agent framework

## Project Shape

```text
artifacts/
  datasets/
  memory/memory.json
  models/
  vectorstores/
notebooks/
  finetuning/
src/dealsight_intelligence/
  agents/
  app/
  data/
  evaluation/
  modal/
  pricing/
scripts/
tests/
```

The Python package is named `dealsight_intelligence` for import stability; the user-facing project name is **DealSight Intelligence**.

## Quick Start

Create an environment and install the package:

```powershell
cd price-is-right
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[app,dev]"
copy .env.example .env
```

## Dataset Sources

Default local-development app dataset:

```text
abhinavvathadi/items_lite
```

Dataset purpose matters:

```text
items_lite / items_full
  Main structured curated datasets. Use these for the app, vector store,
  DNN train/eval, and pricing pipelines.

items_prompts_lite / items_prompts_full
  Prompt/completion datasets. Use these only for prompt-based fine-tuning
  or prompt-evaluation scripts. Do not use them for vector stores.

items_raw_lite / items_raw_full
  Rawer source datasets. They are supported as sources, but are not the
  default app-ready datasets.
```

Supported Hugging Face sources:

```text
abhinavvathadi/items_raw_full
abhinavvathadi/items_full
abhinavvathadi/items_prompts_full
abhinavvathadi/items_lite
abhinavvathadi/items_prompts_lite
abhinavvathadi/items_raw_lite
```

You can also use local exported Hugging Face dataset folders by passing their local path to `--dataset`.

Download the default structured app dataset:

```powershell
python scripts/01_curate_lite.py --from-source
```

Equivalent explicit command:

```powershell
python scripts/01_curate_lite.py --from-source --purpose structured --dataset abhinavvathadi/items_lite --prefix lite
```

Create a 20k lite dataset from raw Amazon metadata instead:

```powershell
python scripts/01_curate_lite.py --train-size 20000 --test-size 2000
```

Switch to your full structured dataset:

```powershell
python scripts/01_curate_lite.py --from-source --purpose structured --dataset abhinavvathadi/items_full --prefix full
python scripts/02_build_vectorstore.py --dataset-path artifacts\datasets\train_full.pkl
```

Export prompt/completion data only for prompt training or prompt eval:

```powershell
python scripts/01_curate_lite.py --from-source --purpose prompt --dataset abhinavvathadi/items_prompts_lite --prefix prompts_lite
```

## Fine-Tuning Pipeline

The QLoRA fine-tuning notebooks for the specialist pricing model live in:

```text
notebooks/finetuning/
```

They cover LoRA/QLoRA setup, base-model evaluation, lite-mode training on `items_prompts_lite`, and testing the resulting PEFT adapter. These notebooks document the fine-tuning data path only; the runtime app and vector store continue to use structured `items_lite` data.

Configure the same defaults in `.env`:

```text
dealsight_intelligence_STRUCTURED_DATASET_SOURCE=abhinavvathadi/items_lite
dealsight_intelligence_DATASET_PREFIX=lite
dealsight_intelligence_PROMPT_DATASET_SOURCE=abhinavvathadi/items_prompts_lite
dealsight_intelligence_RAW_DATASET_SOURCE=abhinavvathadi/items_raw_lite
```

Run one dry-run planning cycle:

```powershell
python scripts/05_run_app.py --once
```

Use the trained deep neural network weights:

```powershell
copy <path-to>\deep_neural_network.pth artifacts\models\deep_neural_network.pth
python -m pip install -e ".[dnn]"
```

Optional W&B experiment tracking is included for retraining/evaluating the DNN with metrics:

```powershell
python -m pip install -e ".[dnn,tracking]"
python scripts/06_train_dnn_wandb.py --wandb --limit 1000
python scripts/07_evaluate_dnn_wandb.py --wandb --limit 200
```

W&B is disabled by default. To run without W&B, omit `--wandb`:

```powershell
python scripts/06_train_dnn_wandb.py --limit 1000
python scripts/07_evaluate_dnn_wandb.py --limit 200
```

To keep W&B local/offline when enabled:

```powershell
$env:WANDB_MODE="offline"
```

Run the Gradio app:

```powershell
python scripts/05_run_app.py
```

Dry-run mode is on by default. Add API keys and set these values when you are ready for live integrations:

```text
dealsight_intelligence_DRY_RUN=false
dealsight_intelligence_DO_PUSH=true
OPENAI_API_KEY=...
PUSHOVER_USER=...
PUSHOVER_TOKEN=...
```

## Architecture

| Agent | Role |
|------|------|
| Scanner | Pulls DealNews RSS feeds, filters out sale events and category pages, summarizes individual product deals |
| Frontier | GPT-4o-mini with vector-store retrieval to price the product against similar items |
| Specialist | Modal-hosted QLoRA fine-tuned Llama adapter for a domain-tuned price estimate |
| Neural Network | Local deep residual MLP on hashed text features |
| Ensemble | Weighted blend of the three pricing signals |
| Planning | Picks the best opportunity by discount and triggers messaging |
| Messaging | Pushover alert (dry-run by default) |

## Roadmap

1. **v1** — RSS scrape, heuristic or OpenAI scan, local pricing fallback, memory, dry-run alerts, Gradio table.
2. **Better pricing** — create/download the lite dataset, build the vector store, enable RAG-based frontier pricing.
3. **Specialist DNN path** — drop `deep_neural_network.pth` into `artifacts/models/` and install the `dnn` extra.
4. **Optional tracking** — use W&B scripts for DNN train/validation/test metrics.
5. **Full deployment** — deploy the Modal `Pricer`, train `ensemble_model.pkl`, enable Pushover, plot the 3D vector-store embedding view.
