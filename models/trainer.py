from transformers import Trainer, get_linear_schedule_with_warmup, TrainerCallback
from torch.utils.data import DataLoader
import torch
from dataset import custom_collate_fn
import torch.nn as nn

# def logits_to_one_hot(predictions, threshold=0.5):
#     reshaped_preds = predictions.reshape(-1, 3, 3)
    
#     one_hot_preds = (reshaped_preds > threshold).int()

#     return one_hot_preds
def logits_to_one_hot(logits):
    probabilities = torch.sigmoid(logits)
    return (probabilities > 0.5).int()  # using 0.5 as threshold


class CustomTrainer(Trainer):
    def __init__(self, learning_rate, weight_decay, class_weights, *args, **kwargs):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        super().__init__(*args, **kwargs)
        self.lr_scheduler = None  # init lr_scheduler
        self.optimizer = None
        print(f"did we took class_weights correctly: {class_weights}")
        if class_weights is not None:
            self.class_weights = class_weights
        else:
            self.class_weights = None
        self.criterion = nn.BCEWithLogitsLoss() # should we use sigmoid version ???? if yes why ???
        
    def create_optimizer(self):
        if self.optimizer is None:
            # Create optimizer
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.003, weight_decay=0.0001) # default lr and weightdecay
        return self.optimizer

    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # print(labels.shape)
        # print(f"labels: {labels}")
        # print(logits.shape)
        # print(f"logits: {logits}")
        total_loss = 0
        for chnl in range(3):
            total_loss += self.criterion(logits.permute(1,0)[chnl], labels.squeeze().permute(1,0)[chnl].float())
        total_loss = total_loss / 3
        print(f"loss: {total_loss}")
        if return_outputs:
            return total_loss, outputs
        return total_loss
    
    
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
    
    #NOTE below is deactivated due to some problems
    # def get_train_dataloader(self) -> DataLoader:
    #     if self.train_dataset is None:
    #         raise ValueError("Trainer: training requires a train_dataset.")
    #     train_sampler = self._get_train_sampler()

    #     return DataLoader(
    #         self.train_dataset,
    #         batch_size=self.args.train_batch_size,
    #         sampler=train_sampler,
    #         collate_fn=custom_collate_fn,
    #         # drop_last=self.args.dataloader_drop_last,
    #         # num_workers=self.args.dataloader_num_workers,
    #     )

    # def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
    #     if eval_dataset is None and self.eval_dataset is None:
    #         raise ValueError("Trainer: evaluation requires an eval_dataset.")
    #     eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

    #     return DataLoader(
    #         eval_dataset,
    #         batch_size=self.args.eval_batch_size,
    #         sampler=self._get_eval_sampler(eval_dataset),
    #         collate_fn=custom_collate_fn,
    #         # drop_last=self.args.dataloader_drop_last,
    #         # num_workers=self.args.dataloader_num_workers,
    #     )
