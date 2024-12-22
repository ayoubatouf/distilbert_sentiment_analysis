from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from inference.predictor import Predictor

app = FastAPI(title="Sentiment Analysis API", version="1.0")


class SentimentRequest(BaseModel):
    text: str


model_dir = "../distilbert-imdb"
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

predictor = Predictor()
predictor.initialize(model, tokenizer)


@app.post("/predict")  # ensure enable GPU for fast inference
def predict_sentiment(request: SentimentRequest):
    try:
        sentiment = predictor.predict(request.text)
        return {"text": request.text, "sentiment": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
