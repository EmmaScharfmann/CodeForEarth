class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def fit(self, train_data, val_data=None):
        raise NotImplemented()
