from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, VivitConfig, VivitForVideoClassification, VivitImageProcessor
import pytorchvideo.data as pd_video
from dataset import read_video_pyav, sample_all_frame_indices, VideoLabelDataset
from transformers import TrainingArguments, Trainer, get_scheduler, set_seed
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Subset
from trainer import CustomTrainer
import evaluate
import av
from datasets import DatasetBuilder, Dataset, load_dataset, load_metric
from torchvision import transforms
import logging

logging.basicConfig(level=logging.INFO)

# "BUY":0,
# "HOLD":1,
# "SELL":2
ckpt = "google/vivit-b-16x2-kinetics400"
def load_dataset_from_txt(path):
    videos = []
    labels = []
    with open(path) as f:
        lines = f.readlines()
        for l in lines:
            video, label = l.strip().split('\t')[0], l.strip().split('\t')[1]
            if label == 'BUY':
                label = 0
            elif label == 'SELL':
                label = 2
            else:
                label = 1
            videos.append(video)
            labels.append(label)
    return videos, labels

def create_csv(videos, labels, dataset_dir):
    df = pd.DataFrame({
        'video': videos,
        'label': labels
    })
    df.to_csv(f"{dataset_dir}/dataset.csv")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Transform pipeline
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
])


train_path = "/home/emir/Desktop/dev/datasets/ETF_RGB_Videos/labels.txt"
test_path = "/home/emir/Desktop/dev/datasets/ETF_Video_Test/labels.txt"
dataset_rooth_path = "/home/emir/Desktop/dev/datasets/ETF_RGB_Videos/"
train_videos, train_labels = load_dataset_from_txt(train_path)
test_videos, test_labels = load_dataset_from_txt(test_path)



train_dataset = VideoLabelDataset("/home/emir/Desktop/dev/datasets/ETF_RGB_Videos/dataset.csv", transform=train_transform)
val_test_dataset = VideoLabelDataset("/home/emir/Desktop/dev/datasets/ETF_Video_Test/dataset.csv", transform=val_transform)
split_idx = int(len(val_test_dataset) / 2)
val_dataset = Subset(val_test_dataset, range(0, split_idx))
test_dataset = Subset(val_test_dataset, range(split_idx, len(val_test_dataset)))
# print(len(val_test_dataset))
# print(len(test_dataset))
# print(len(val_dataset))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return load_metric("f1").compute(predictions=predictions, references=labels, average="micro")



vivit_cfg = VivitConfig(
    image_size=64,
    num_frames=64,
    patch_size=4,
    num_labels=3,
    hidden_size=768,        # Reduced from default (e.g., 768)
    num_hidden_layers=12,    # Reduced from default (e.g., 12)
    num_attention_heads=12,  # Reduced from default (e.g., 12)
    # intermediate_size=2048, # Adjust as needed, default might be higher
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)

model = VivitForVideoClassification.from_pretrained(ckpt, config=vivit_cfg, ignore_mismatched_sizes=True)

set_seed(42)

total_num_samples = len(train_dataset)
batch_size = 4
steps_per_epoch = total_num_samples // batch_size

training_args = TrainingArguments(
    output_dir="./model_output",
    num_train_epochs=100,
    per_device_train_batch_size=batch_size,
    learning_rate=1e-5,  # Reduced learning rate
    per_device_eval_batch_size=batch_size,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    save_strategy='epoch',
    load_best_model_at_end=True,
    save_total_limit=5,
    max_grad_norm=1.0  # Gradient clipping
)


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
