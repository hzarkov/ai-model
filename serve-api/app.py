import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/flan-t5-large-finetuned")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
model = None

def load_model():
    global tokenizer, model
    print(f"[DEBUG] MODEL_PATH: {MODEL_PATH}")
    print(f"[DEBUG] Directory contents: {os.listdir(MODEL_PATH)}")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE)

load_model()

app = FastAPI()

SYSTEM_PROMPT = (
    "Extract all time points, activities, and food items from the following note. "
    "Return the result as a JSON list of events, each with a \"timestamp\" (if available), \"label\", and \"details\" (if available). "
    "For food, list all main ingredients as tokens in a \"food\" field.\n\n"
    "Example:\n"
    "Note: 10:30UTC - Took me 20minutes to do my lunch (chicken with rice), but now it's ready and I'm going to eat it for 20 min so I have time for a short desert(5min) before my next work meeting\n\n"
    "Output:\n"
    "[\n"
    "  {\"timestamp\": \"10:10UTC\", \"label\": \"start_lunch_prep\"},\n"
    "  {\"timestamp\": \"10:30UTC\", \"label\": \"lunch_ready\", \"food\": [\"chicken\", \"rice\"]},\n"
    "  {\"timestamp\": \"10:30UTC\", \"label\": \"start_eating\"},\n"
    "  {\"timestamp\": \"10:50UTC\", \"label\": \"start_desert\"},\n"
    "  {\"timestamp\": \"10:55UTC\", \"label\": \"back_to_work\"}\n"
    "]\n\n"
    "Note:"
)

@app.post("/generate")
def generate(input_text: str):
    try:
        prompt = f"{SYSTEM_PROMPT}\n{input_text}"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=128,
                temperature=1,
            )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "FLAN-T5 FastAPI server is running."}
