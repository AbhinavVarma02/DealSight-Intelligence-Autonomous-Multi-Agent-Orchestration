"""Modal cloud service that serves the QLoRA fine-tuned specialist pricer.

The Specialist agent calls this service over the network. The class
loads a 4-bit quantised base Llama model plus the PEFT adapter at
container start, then answers `price(description)` requests by
generating a few tokens and parsing the resulting dollar amount.
"""

from __future__ import annotations

import os

import modal
from modal import Image, Volume

# Defaults below point at a publicly available adapter. Override via env
# vars if you have fine-tuned your own.

app = modal.App("pricer-service")

image = Image.debian_slim().pip_install(
    "huggingface",
    "torch",
    "transformers",
    "bitsandbytes",
    "accelerate",
    "peft",
)

# Depending on your Modal setup, the secret may be named "hf-secret" or
# "huggingface-secret". Override locally before deploy if needed:
# $env:DEALSIGHT_INTELLIGENCE_MODAL_HF_SECRET="huggingface-secret"
HF_SECRET_NAME = os.getenv("DEALSIGHT_INTELLIGENCE_MODAL_HF_SECRET", "hf-secret")
secrets = [modal.Secret.from_name(HF_SECRET_NAME)]

GPU = os.getenv("DEALSIGHT_INTELLIGENCE_MODAL_GPU", "T4")
CACHE_DIR = "/cache"
MIN_CONTAINERS = int(os.getenv("DEALSIGHT_INTELLIGENCE_MODAL_MIN_CONTAINERS", "0"))

# Default specialist adapter. Replace via env vars if you have your own.
BASE_MODEL = os.getenv("DEALSIGHT_INTELLIGENCE_BASE_MODEL", "meta-llama/Meta-Llama-3.1-8B")
PROJECT_NAME = os.getenv("DEALSIGHT_INTELLIGENCE_FINETUNE_PROJECT_NAME", "pricer")
HF_USER = os.getenv("DEALSIGHT_INTELLIGENCE_FINETUNE_HF_USER", "ed-donner")
RUN_NAME = os.getenv("DEALSIGHT_INTELLIGENCE_FINETUNE_RUN_NAME", "2024-09-13_13.04.39")
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
REVISION = os.getenv("DEALSIGHT_INTELLIGENCE_FINETUNE_REVISION", "e8d637df551603dc86cd7a1598a8f44af4d7ae36")
FINETUNED_MODEL = os.getenv("DEALSIGHT_INTELLIGENCE_FINETUNED_MODEL", f"{HF_USER}/{PROJECT_RUN_NAME}")

QUESTION = "How much does this cost to the nearest dollar?"
PREFIX = "Price is $"

hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.cls(
    image=image.env({"HF_HUB_CACHE": CACHE_DIR}),
    secrets=secrets,
    gpu=GPU,
    timeout=1800,
    min_containers=MIN_CONTAINERS,
    volumes={CACHE_DIR: hf_cache_volume},
)
class Pricer:
    @modal.enter()
    def setup(self):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            device_map="auto",
        )
        self.fine_tuned_model = PeftModel.from_pretrained(
            self.base_model,
            FINETUNED_MODEL,
            revision=REVISION,
        )

    @modal.method()
    def price(self, description: str) -> float:
        import re
        import torch
        from transformers import set_seed

        set_seed(42)
        prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        attention_mask = torch.ones(inputs.shape, device="cuda")
        with torch.no_grad():
            outputs = self.fine_tuned_model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=5,
                num_return_sequences=1,
            )
        result = self.tokenizer.decode(outputs[0])
        contents = result.split(PREFIX)[1].replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
        return float(match.group()) if match else 0.0


def deployed_model_details() -> dict[str, str]:
    return {
        "base_model": BASE_MODEL,
        "finetuned_model": FINETUNED_MODEL,
        "revision": REVISION,
        "modal_app": "pricer-service",
        "modal_class": "Pricer",
    }
