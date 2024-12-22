class DatasetTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize_dataset(self, dataset, text_col="review", label_col="sentiment"):
        def tokenize_function(examples):
            outputs = self.tokenizer(examples[text_col], truncation=True, padding=True)
            outputs["label"] = examples[label_col]
            return outputs

        return dataset.map(tokenize_function, batched=True)
