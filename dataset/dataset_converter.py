from datasets import Dataset


class DatasetConverter:
    @staticmethod
    def convert_to_dataset(df):
        return Dataset.from_pandas(df)
