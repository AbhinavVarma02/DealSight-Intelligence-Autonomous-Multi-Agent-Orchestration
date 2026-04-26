---
title: DealSight Intelligence
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# DealSight Intelligence

## Autonomous Multi-Agent Orchestration for Real-Time Deal Discovery

DealSight Intelligence is a production-style AI system that scans live e-commerce deal feeds, estimates product value through a multi-agent pricing ensemble, and surfaces high-discount opportunities through a real-time dashboard and alerting workflow.

Live demo: [DealSight Intelligence on Hugging Face Spaces](https://abhinavvathadi-dealsight-intelligence.hf.space/)

The system combines retrieval-augmented generation, a QLoRA fine-tuned Llama specialist, and a local PyTorch residual neural network to estimate whether a product is meaningfully underpriced compared with similar historical listings.

## Public Demo Note

The Hugging Face Space is deployed in dry-run demo mode so recruiters and reviewers can understand the application flow without exposing private API keys or triggering paid external services. In this mode, live alerts are disabled and optional paid integrations such as OpenAI, Modal, Pushover, and hosted specialist inference can be enabled only through private environment secrets.

## Project Summary

This project was designed as an end-to-end applied AI system for deal intelligence. It connects live RSS deal feeds, product retrieval, fine-tuned model inference, neural network pricing, ensemble scoring, memory tracking, and user-facing alerts.

The core pipeline:

1. Scans live e-commerce RSS feeds.
2. Extracts candidate product deals.
3. Retrieves the top 5 similar products from a 20,000-document Chroma vector store.
4. Generates price estimates from three independent agents.
5. Blends estimates using an 80/10/10 weighting strategy.
6. Surfaces high-discount deals through a Gradio dashboard and optional Pushover alerts.

## Key Highlights

- Fine-tuned Meta Llama 3.2-3B on 20,000 e-commerce product listings using QLoRA.
- Reduced GPU memory usage from 6.4 GB to 2.2 GB with 4-bit NF4 quantization.
- Produced a compact 73.4 MB PEFT adapter for specialist price estimation.
- Deployed the fine-tuned specialist model as a serverless GPU inference endpoint on Modal.
- Tracked fine-tuning experiments and training metrics with Weights & Biases.
- Built a 3-agent pricing ensemble using GPT-4o-mini with RAG, a QLoRA fine-tuned Llama specialist, and a PyTorch residual DNN.
- Used a 20,000-document Chroma vector store for similar-product retrieval.
- Built a real-time Gradio dashboard for deal review.
- Added Pushover-based alerting for high-discount opportunities.
- Designed safe fallback behavior so the system can run even when optional services are unavailable.

## System Architecture

| Component | Purpose |
|---|---|
| Scanner Agent | Reads live e-commerce RSS feeds and extracts candidate product deals |
| Frontier Agent | Uses GPT-4o-mini with retrieval-augmented context from similar products |
| Specialist Agent | Calls a QLoRA fine-tuned Llama model served through Modal |
| Neural Network Agent | Uses a local PyTorch residual DNN for offline price estimation |
| Ensemble Agent | Blends pricing estimates using an 80/10/10 weighting strategy |
| Planning Agent | Selects the strongest deal based on estimated discount |
| Messaging Agent | Sends optional Pushover alerts, with dry-run mode enabled by default |
| Gradio App | Provides a real-time interface for reviewing discovered deals |

## Technical Stack

| Area | Tools and Libraries |
|---|---|
| Language | Python |
| LLMs | GPT-4o-mini, Meta Llama 3.2-3B |
| Fine-tuning | QLoRA, PEFT, 4-bit NF4 quantization |
| Model Hosting | Modal serverless GPU endpoint |
| Experiment Tracking | Weights & Biases |
| Retrieval | Chroma vector store |
| Deep Learning | PyTorch, torch.nn, CosineAnnealingLR, gradient clipping |
| Dashboard | Gradio |
| Alerts | Pushover |
| Testing | Pytest |
| Packaging | pyproject.toml, editable install |

## Repository Structure

```text
artifacts/
  datasets/
  memory/
  models/
  vectorstores/

notebooks/
  finetuning/

scripts/
  01_curate_lite.py
  02_build_vectorstore.py
  04_train_ensemble.py
  05_run_app.py
  06_train_dnn_wandb.py
  07_evaluate_dnn_wandb.py

src/
  dealsight_intelligence/
    agents/
    app/
    data/
    evaluation/
    modal/
    pricing/

tests/
```

## Quick Start

Clone the repository and create a virtual environment:

```powershell
git clone https://github.com/AbhinavVarma02/DealSight-Intelligence-Autonomous-Multi-Agent-Orchestration.git
cd DealSight-Intelligence-Autonomous-Multi-Agent-Orchestration

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[app,dev]"
copy .env.example .env
```

Run one dry-run planning cycle:

```powershell
python scripts/05_run_app.py --once
```

Run the Gradio app:

```powershell
python scripts/05_run_app.py
```

Dry-run mode is enabled by default, so the app can be tested safely without sending real alerts.

## Environment Variables

Create a local `.env` file from `.env.example`.

Required only for live integrations:

```text
OPENAI_API_KEY=
PUSHOVER_USER=
PUSHOVER_TOKEN=
MODAL_TOKEN_ID=
MODAL_TOKEN_SECRET=
WANDB_API_KEY=
```

Example runtime settings:

```text
DEALSIGHT_INTELLIGENCE_DRY_RUN=true
DEALSIGHT_INTELLIGENCE_DO_PUSH=false
DEALSIGHT_INTELLIGENCE_STRUCTURED_DATASET_SOURCE=abhinavvathadi/items_lite
DEALSIGHT_INTELLIGENCE_DATASET_PREFIX=lite
DEALSIGHT_INTELLIGENCE_PROMPT_DATASET_SOURCE=abhinavvathadi/items_prompts_lite
DEALSIGHT_INTELLIGENCE_RAW_DATASET_SOURCE=abhinavvathadi/items_raw_lite
```

Keep real API keys only in your local `.env` file. Do not commit `.env`.

## Dataset Sources

The project supports structured, prompt, and raw dataset variants.

Default structured dataset:

```text
abhinavvathadi/items_lite
```

Supported Hugging Face datasets:

```text
abhinavvathadi/items_raw_full
abhinavvathadi/items_full
abhinavvathadi/items_prompts_full
abhinavvathadi/items_lite
abhinavvathadi/items_prompts_lite
abhinavvathadi/items_raw_lite
```

Dataset usage:

| Dataset Type | Purpose |
|---|---|
| items_lite or items_full | App runtime, vector store, DNN training, pricing pipelines |
| items_prompts_lite or items_prompts_full | Prompt-based fine-tuning and prompt evaluation |
| items_raw_lite or items_raw_full | Raw source data for curation |

Download the default structured dataset:

```powershell
python scripts/01_curate_lite.py --from-source
```

Build the vector store:

```powershell
python scripts/02_build_vectorstore.py --dataset-path artifacts\datasets\train_lite.pkl
```

Create a 20,000-product curated dataset from raw metadata:

```powershell
python scripts/01_curate_lite.py --train-size 20000 --test-size 2000
```

## Fine-Tuning Workflow

Fine-tuning notebooks are located in:

```text
notebooks/finetuning/
```

The notebooks cover:

1. QLoRA and quantization setup.
2. Baseline model evaluation.
3. Fine-tuning Meta Llama 3.2-3B on product listing data.
4. Evaluation of the fine-tuned PEFT adapter.

The fine-tuned specialist model is designed to provide a domain-specific pricing estimate inside the larger ensemble.

## Deep Neural Network Workflow

Train the PyTorch residual DNN with optional Weights & Biases tracking:

```powershell
python -m pip install -e ".[dnn,tracking]"
python scripts/06_train_dnn_wandb.py --wandb --limit 1000
python scripts/07_evaluate_dnn_wandb.py --wandb --limit 200
```

Run without Weights & Biases:

```powershell
python scripts/06_train_dnn_wandb.py --limit 1000
python scripts/07_evaluate_dnn_wandb.py --limit 200
```

Use offline W&B mode:

```powershell
$env:WANDB_MODE="offline"
```

## Testing

Run the test suite:

```powershell
pytest
```

The tests cover core behavior for datasets, deal extraction, scanner logic, agent framework behavior, and neural network agent functionality.

## Safety and Fallback Design

The system is designed to avoid hard failures when optional services are unavailable.

Examples:

- If Modal is unavailable, the specialist model path can fall back safely.
- If Pushover is not configured, alerts remain in dry-run mode.
- If trained model artifacts are missing, the app can still run with available pricing paths.
- If API keys are missing, live integrations remain disabled.

## Current Status

Implemented:

- Live RSS deal scanning
- Retrieval-augmented pricing
- QLoRA fine-tuning workflow
- Modal specialist endpoint path
- PyTorch residual DNN pricing path
- Weighted 3-agent ensemble
- Gradio dashboard
- Pushover alert workflow
- Unit tests for core modules
- Safe local development setup

## Resume Alignment

This repository supports the following project summary:

Built a production-style multi-agent deal intelligence system combining GPT-4o-mini with RAG, a QLoRA fine-tuned Llama specialist, and a PyTorch residual DNN. The system scans live e-commerce RSS feeds, retrieves similar products from a 20,000-document Chroma vector store, blends pricing estimates, and surfaces high-discount opportunities through a Gradio dashboard and Pushover alerts.

## License

This project is intended for portfolio and educational use.
