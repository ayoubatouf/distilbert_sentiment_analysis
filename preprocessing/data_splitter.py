from sklearn.model_selection import train_test_split


class DataSplitter:
    def __init__(self, test_size=0.2, random_state=42, stratify_col=None):
        self.test_size = test_size
        self.random_state = random_state
        self.stratify_col = stratify_col

    def split(self, df, target_col):
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.stratify_col,
        )
        return X_train, X_test, y_train, y_test
