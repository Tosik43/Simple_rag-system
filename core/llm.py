import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from core.config import LLM_MODEL

print("[LLM] Загрузка модели...")

DEVICE = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    dtype=torch.float32,
    trust_remote_code=True,
)

model.to(DEVICE)
model.eval()

print("[LLM] Модель загружена на CPU.")