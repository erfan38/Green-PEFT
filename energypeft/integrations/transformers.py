# energypeft/integrations/transformers.py
from transformers import Trainer, TrainingArguments

class TransformersTrainer:
    def __init__(self, model, train_dataset, eval_dataset=None, training_args=None):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_args = training_args or TrainingArguments(output_dir="./results", num_train_epochs=1)
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )

    def train(self):
        return self.trainer.train()

    def evaluate(self):
        return self.trainer.evaluate()
