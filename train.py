import os
from dataset.dataset_converter import DatasetConverter
from dataset.tokenizer import DatasetTokenizer
from loaders.data_loader import DataLoader
from models.model_initializer import ModelInitializer
from preprocessing.data_splitter import DataSplitter
from preprocessing.sentiment_mapper import SentimentMapper
from training.trainer import ModelTrainer
from training.training_config import TrainingConfig


def main():  # ensure enable GPU for fast training

    data_loader = DataLoader(file_path="/raw/IMDB Dataset.csv")
    df = data_loader.load_data()

    data_splitter = DataSplitter(stratify_col=df["sentiment"])
    X_train, X_test, y_train, y_test = data_splitter.split(df, target_col="sentiment")

    train_df = X_train.copy()
    train_df["sentiment"] = y_train
    test_df = X_test.copy()
    test_df["sentiment"] = y_test

    sentiment_mapper = SentimentMapper({"negative": 0, "positive": 1})
    train_df = sentiment_mapper.map_sentiments(train_df, "sentiment")
    test_df = sentiment_mapper.map_sentiments(test_df, "sentiment")

    initializer = ModelInitializer()
    model, tokenizer = initializer.initialize_model_and_tokenizer()

    dataset_converter = DatasetConverter()
    train_ds = dataset_converter.convert_to_dataset(train_df)
    test_ds = dataset_converter.convert_to_dataset(test_df)

    dataset_tokenizer = DatasetTokenizer(tokenizer)
    tokenized_train_ds = dataset_tokenizer.tokenize_dataset(train_ds)
    tokenized_test_ds = dataset_tokenizer.tokenize_dataset(test_ds)

    training_args = TrainingConfig.get_training_args()
    compute_metrics = TrainingConfig.compute_metrics

    trainer = ModelTrainer(model, tokenizer, training_args)
    trainer.train(tokenized_train_ds, tokenized_test_ds, compute_metrics)

    output_dir = "sentiment_model"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")


if __name__ == "__main__":
    main()
