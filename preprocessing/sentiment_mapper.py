class SentimentMapper:
    def __init__(self, mapping):
        self.mapping = mapping

    def map_sentiments(self, df, target_col):
        df[target_col] = df[target_col].map(self.mapping)
        return df
