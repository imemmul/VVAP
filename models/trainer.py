from transformers import Trainer, AdamW, get_linear_schedule_with_warmup, TrainerCallback


class CustomTrainer(Trainer):
    def __init__(self, learning_rate, weight_decay, *args, **kwargs):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        super().__init__(*args, **kwargs)
        self.lr_scheduler = None  # init lr_scheduler
        self.optimizer = None

    def create_optimizer(self):
        if self.optimizer is None:
            # Create optimizer
            self.optimizer = AdamW(self.model.parameters(), lr=1e-5, weight_decay=0.01)
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if optimizer is None:
            optimizer = self.create_optimizer()

        if self.lr_scheduler is None:
            # Create scheduler
            self.lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps
            )
        return self.lr_scheduler
