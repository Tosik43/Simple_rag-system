from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Название модели
model_name = "Qwen/Qwen3-8b"   

# Путь сохранения
save_dir = "models/qwen3-8b"

print(f"Скачиваем модель {model_name}")
print(f"Сохраняем в: {save_dir}")

# Скачиваем
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",           # сначала скачиваем на CPU
    torch_dtype="auto",
    trust_remote_code=True,
    # load_in_8bit=True,        # можно включить позже
)

# Сохраняем в нашу папку
print("Сохраняем модель локально...")
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"✅ Модель успешно сохранена в папку: {save_dir}")