from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ModelInitializer:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name

    def initialize_model_and_tokenizer(self):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer
