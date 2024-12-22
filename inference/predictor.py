import torch


class Predictor:
    _instance = None  # store the singleton instance

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
            cls._instance.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        return cls._instance

    def initialize(self, model, tokenizer):
        if self.model is None or self.tokenizer is None:
            self.model = model
            self.tokenizer = tokenizer
            self.model.to(self.device)

    def predict(self, review):
        if not self.model or not self.tokenizer:
            raise RuntimeError("predictor is not initialized. call 'initialize' first")

        inputs = self.tokenizer(
            review, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

        return "positive" if predicted_class == 1 else "negative"
