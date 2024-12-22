import numpy as np
import evaluate  # type: ignore
from transformers import TrainingArguments


class TrainingConfig:
    @staticmethod
    def compute_metrics(eval_preds):
        metric = evaluate.load("accuracy")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    @staticmethod
    def get_training_args(output_dir="distilbert-imdb"):
        return TrainingArguments(
            num_train_epochs=1,
            output_dir=output_dir,
            report_to="wandb",
            push_to_hub=False,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            evaluation_strategy="epoch",
        )
