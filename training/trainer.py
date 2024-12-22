from transformers import Trainer, DataCollatorWithPadding


class ModelTrainer:
    def __init__(self, model, tokenizer, training_args):
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.data_collator = DataCollatorWithPadding(tokenizer)

    def train(self, train_dataset, eval_dataset, compute_metrics):
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        return trainer
