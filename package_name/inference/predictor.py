class Model:
    def __init__(self, model):
        self.model = model

    def predict(self, inputs: list) -> list:
        raise NotImplementedError()
