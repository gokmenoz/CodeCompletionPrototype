import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- CONFIG ---
MODEL_PATH = "analysis_output/local_codellama_model"
LORA_PATH = "analysis_output/lora_adapters"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    return model.to(DEVICE), tokenizer

model, tokenizer = load_model()

# --- UI ---
st.title("💡 CodeLLaMA Code Completion Prototype")
prompt = st.text_area("✍️ Enter a Python function prompt", value="def add(a, b):", height=100)

col1, col2 = st.columns(2)
with col1:
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
    top_p = st.slider("Top-p (nucleus sampling)", 0.1, 1.0, 0.95, 0.05)
with col2:
    top_k = st.slider("Top-k", 0, 100, 50, 5)
    max_new_tokens = st.slider("Max new tokens", 16, 256, 100, 8)

if st.button("🚀 Generate Completion"):
    with st.spinner("Generating..."):
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )

        completion = tokenizer.decode(output[0], skip_special_tokens=True)

        st.subheader("✅ Completion")
        st.code(completion[len(prompt):], language="python")
