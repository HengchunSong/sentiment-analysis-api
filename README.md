# Sentiment Analysis API with LangChain + Ollama (Llama 3.2 3B)

A FastAPI web service that analyzes text sentiment using **Llama 3.2 3B** running locally via Ollama.

### Features
- POST `/analyze-sentiment` → returns JSON with `sentiment` and natural-language `explanation`
- Uses local Ollama (no API keys, fully offline after first load)
- LangChain v1.0+ with LCEL
- Containerized with Docker/Podman
- Swagger UI at http://localhost:8000/docs

### Requirements
- Ollama installed and running (`ollama serve`)
- Model pulled: `ollama pull llama3.2:3b`

### Local run (no Docker)
```bash
pip install -r requirements.txt
uvicorn llm:app --reload"
Docker/Podman run
Bash# Make Ollama listen on all interfaces
export OLLAMA_HOST=0.0.0.0

# Build and run
docker build -t localhost/sentiment-api:latest .
docker run -d -p 8000:8000 --name sentiment-api localhost/sentiment-api:latest
Example
Bashcurl -X POST http://localhost:8000/analyze-sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this project!"}'
Response:
JSON{
  "sentiment": "positive",
  "explanation": "The user expresses strong affection using the word 'love' and enthusiasm about the project."
}
Project completed November 19, 2025
Built with FastAPI • LangChain • Ollama • Llama 3.2 3B • Podman
