# Specialist Fine-Tuning Notebooks

These notebooks document the QLoRA fine-tuning pipeline used to produce the specialist pricing model that DealSight Intelligence calls through Modal.

```text
01_qlora_quantization_explorer.ipynb
  Introduces LoRA and QLoRA, compares quantization modes, and shows the lite-mode LoRA setup.

02_baseline_evaluation.ipynb
  Evaluates the base Llama model against the price-prediction prompt dataset.

03_qlora_finetuning_pipeline.ipynb
  Runs lite-mode QLoRA fine-tuning with `LITE_MODE=True`, `items_prompts_lite`, W&B tracking, and Hugging Face Hub uploads.

04_finetuned_model_evaluation.ipynb
  Loads the PEFT adapter and evaluates the fine-tuned model on the test split.
```

Important distinction:

```text
items_prompts_lite
  Prompt/completion data for fine-tuning and prompt evaluation.

items_lite
  Structured app data for vector stores, DNN training/evaluation, and runtime pricing.
```

The notebooks are not required to run the local DealSight Intelligence app. They are kept here as the provenance trail for how the lite-mode fine-tuning data and PEFT adapter were produced.
