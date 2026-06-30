from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import os

def load_local_model():
    model_path = "models/qwen3-8b"
    
    print(f"Загрузка модели из {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",           # автоматически использует GPU
        torch_dtype=torch.float16,
        trust_remote_code=True,
        load_in_8bit=True,           
        # load_in_4bit=True,         
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.2,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    print("✅ Локальная модель успешно загружена!")
    return llm