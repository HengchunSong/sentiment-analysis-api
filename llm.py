# app.py (Legacy fix for v1.0+)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_classic.chains import LLMChain  # Updated: Use _classic here
import os

app = FastAPI()

# Load HuggingFace API token from environment variable
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable is not set")

# Initialize the LLM (use a generative model like gpt2 for explanations)
llm = HuggingFaceHub(
    repo_id="gpt2",  # Better for text generation than classification models
    huggingfacehub_api_token=hf_token,
    model_kwargs={"temperature": 0.5, "max_length": 150}
)

# Define the prompt template
prompt = PromptTemplate(
    input_variables=["text"],
    template="Analyze the sentiment of the following text: '{text}'. Classify it as positive, negative, or neutral. Provide a short explanation. Respond in this exact format:\nSentiment: [label]\nExplanation: [reason]"
)

# Create the chain (unchanged)
chain = LLMChain(llm=llm, prompt=prompt)

class TextInput(BaseModel):
    text: str

@app.post("/analyze-sentiment")
async def analyze_sentiment(input: TextInput):
    if not input.text:
        raise HTTPException(status_code=400, detail="Text input is required")
    
    try:
        result = chain.run(input.text)
        # Parse the result based on prompted format
        lines = result.strip().split("\n")
        sentiment = lines[0].replace("Sentiment:", "").strip().lower() if lines else "neutral"
        explanation = " ".join(lines[1:]).replace("Explanation:", "").strip() if len(lines) > 1 else "No explanation provided."
        
        if sentiment not in ["positive", "negative", "neutral"]:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running!"}