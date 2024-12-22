from transformers import AutoModelForSequenceClassification, AutoTokenizer
from inference.predictor import Predictor


def main():  # ensure enable GPU for fast inference

    model_dir = "distilbert-imdb"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    predictor = Predictor()
    predictor.initialize(model, tokenizer)

    new_review = "I didn't enjoy it!"
    sentiment = predictor.predict(new_review)

    print(f"The sentiment of the review is: {sentiment}")


if __name__ == "__main__":
    main()
