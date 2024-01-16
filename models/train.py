from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, VivitConfig, VivitForVideoClassification, VivitImageProcessor
import pytorchvideo.data as pd_video
from dataset import read_video_pyav, sample_all_frame_indices, VideoLabelDataset, custom_collate_fn
# from logi.utils.class_weight import compute_class_weight
from transformers import TrainingArguments, Trainer, get_scheduler, set_seed
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Subset
from trainer import CustomTrainer, logits_to_one_hot
import evaluate
import av
from datasets import DatasetBuilder, Dataset, load_dataset, load_metric
from torchvision import transforms
import logging
import argparse
import os
import ast

logging.basicConfig(level=logging.INFO) #FIXME no logging
#TODO other concern that needs to be done is that is it possible to give data ordered to model ?????
# "BUY":0,
# "HOLD":1, #NOTE right now created dataset is imbalanced so should fix this shit
# "SELL":2
#TODO should need resume
def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_path", type=str, default="/home/emir/Desktop/dev/datasets/ETF_RGB_Videos/labels.txt",
                        # help="Path to the training labels file")
    # parser.add_argument("--test_path", type=str, default="/home/emir/Desktop/dev/datasets/ETF_RGB_Videos_Test/labels.txt",
                        # help="Path to the test labels file")
                    
    parser.add_argument("--train_dataset_root", type=str,
                        help="Root directory of the training dataset")
    parser.add_argument("--test_dataset_root", type=str,
                        help="Root directory of the test dataset")
    parser.add_argument("--output_dir", type=str, default="./model_output",
                        help="Output directory for model checkpoints and logs")
    parser.add_argument("--pretrained_ckpt", type=str, default="google/vivit-b-16x2-kinetics400")
    parser.add_argument("--num_training_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--image_size", type=int, default=64,
                        help="Number of training epochs")
    parser.add_argument("--num_frames", type=int, default=64,
                        help="Number of training epochs")
    parser.add_argument("--patch_size", type=int, default=4, # maybe i can increase patch_size
                        help="Number of training epochs")
    parser.add_argument("--num_labels", type=int, default=3, # classes is like BBS, SBB, BBB 2^3 = 8 ANOTHER APPROACH this is 2 but needs play with loss function in training
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--hidden_size", type=int, default=768,
                        help="Number of training epochs")
    parser.add_argument("--num_hidden_layers", type=int, default=6,
                        help="Number of training epochs")
    parser.add_argument("--num_attention_heads", type=int, default=6,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps for the learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for optimization")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Gradient clipping threshold")
    parser.add_argument("--test", action='store_true',
                        help="Testing")
    parser.add_argument("--generate_csvs", action='store_true',
                        help="csv_generator")
    parser.add_argument("--class_weights", type=list, default=None,
                        help="for imbalance datasets")
    
    args = parser.parse_args()
    return args


def load_dataset_from_txt(path, new_prefix, test):
    """
    this function is a util i know
    """
    videos = []
    labels = []
    with open(os.path.join(path, 'labels.txt')) as f:
        lines = f.readlines()
        for l in lines:
            video, label = l.strip().split('\t')[0], l.strip().split('\t')[1]
            if label == 'BUY':
                label = 0
            elif label == 'SELL':
                label = 2
            else:
                label = 1
            if test:
                video = video.replace("/home/emir/Desktop/dev/datasets/ETF_RGB_Videos_Test/", new_prefix)
            else:
                video = video.replace("/home/emir/Desktop/dev/datasets/ETF_RGB_Videos/", new_prefix)
            videos.append(video)
            labels.append(label)
    return videos, labels

def create_csv(videos, labels, dataset_dir):
    """
    same
    """
    df = pd.DataFrame({
        'video': videos,
        'label': labels
    })
    df.to_csv(f"{dataset_dir}dataset.csv")


def get_datasets(args):
    #NOTE for this task below is not approproiate.
    #NOTE more temporal info focused transforms needed
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = VideoLabelDataset(f"{args.train_dataset_root}labels.csv", transform=train_transform)
    val_dataset = VideoLabelDataset(f"{args.test_dataset_root}labels.csv", transform=val_transform)
    # split_idx = int(len(val_test_dataset) / 2)
    # val_dataset = Subset(val_test_dataset, range(0, split_idx))
    # test_dataset = Subset(val_test_dataset, range(split_idx, len(val_test_dataset)))
    return train_dataset, val_dataset

def create_model(args):
    print(f"creating model: hidden layers: {args.num_hidden_layers}")
    print(f"num_attention_heads: {args.num_attention_heads}")
    vivit_cfg = VivitConfig(
        image_size=args.image_size,
        num_frames=args.num_frames,
        patch_size=args.patch_size,
        num_labels=args.num_labels,
        hidden_size=args.hidden_size,        # (e.g., 768)
        num_hidden_layers=args.num_hidden_layers,    # (e.g., 12)
        num_attention_heads=args.num_attention_heads,  # (e.g., 12)
        # intermediate_size=2048, # Adjust as needed, default might be higher #TODO if still overfit activate this
        hidden_dropout_prob=0.1, # these are fix (e.g. 0.1)
        attention_probs_dropout_prob=0.1
    )

    model = VivitForVideoClassification.from_pretrained(args.pretrained_ckpt, config=vivit_cfg, ignore_mismatched_sizes=True) #TODO check for how to load or keep some weights of mismatched sizes, if possible or is it valid ????? 
    return model

def compute_metrics(eval_pred):
    f1_metric = evaluate.load('f1')
    logits, labels = eval_pred
    logits, labels = torch.tensor(logits), torch.tensor(labels)
    average_f1 = 0
    avg_f1 = {}
    for c in range(3):
        preds = logits_to_one_hot(logits.permute(1,0)[c])
        labels_c = labels.squeeze().permute(1,0)[c]
        average_f1 += f1_metric.compute(predictions=preds, references=labels_c, average='weighted')['f1']
    avg_f1['f1'] = average_f1 / 3
    return avg_f1

def run_train(args):
    set_seed(42)
    model = create_model(args)
    train_dataset, val_dataset = get_datasets(args)
    
    total_num_samples = len(train_dataset)
    batch_size = args.per_device_train_batch_size
    steps_per_epoch = total_num_samples // batch_size #FIXME Training arguments doesn't work like i really wanted i about logging etc.

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_training_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=args.learning_rate, #custom trainer will take this argument
        per_device_eval_batch_size=batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay, # custom trainer will take this argument
        logging_dir='./logs',
        evaluation_strategy="steps",
        save_strategy='steps',
        load_best_model_at_end=True,
        fp16=True,
        # save_total_limit=5,
        logging_steps=500,
        save_steps=500,
        max_grad_norm=args.max_grad_norm, # to test how gradient norm works on like temporal, #CONCLUSION it doesn't effected as much as i expected so check later
        # dataloader_num_workers=1
        eval_steps=500,   
    )
    print(f"setting class_weights: {args.class_weights}")
    trainer = CustomTrainer(
        model=model,
        class_weights=args.class_weights,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
            

def parse_labels(label_str):
    try:
        return ast.literal_eval(label_str)
    except ValueError:
        return []  # or some default value like [0, 0, 0]

def main():
    args = parse_args()
    run_train(args)
if __name__ == "__main__":
    main()