from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json, re

app = FastAPI(title="Sentiment Analysis API with Llama 3.2 3B (Ollama)")

# Connect to Ollama on host (works with --network=host)
llm = ChatOllama(
    model="llama3.2:3b",
    base_url="http://192.168.1.158:11434",
    temperature=0.3
)

# ← THIS IS THE FIXED PROMPT (note the double {{ and }} around the JSON example)
prompt = PromptTemplate.from_template("""
You are a sentiment analysis expert.
Analyze this text and respond with ONLY valid JSON (no markdown, no extra text).

Text: {text}

Response format:
{{
  "sentiment": "positive" | "negative" | "neutral",
  "explanation": "short natural language reason in one sentence"
}}
""")

chain = prompt | llm | StrOutputParser()

class TextInput(BaseModel):
    text: str

@app.post("/analyze-sentiment")
async def analyze_sentiment(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    
    try:
        raw = chain.invoke({"text": input.text})
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        else:
            return {"sentiment": "neutral", "explanation": "Failed to parse response"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Sentiment API running with local Ollama + llama3.2:3b"}
