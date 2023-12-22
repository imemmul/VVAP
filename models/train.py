from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, VivitConfig, VivitForVideoClassification, VivitImageProcessor
import pytorchvideo.data
from dataset import read_video_pyav, sample_all_frame_indices
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
import av
# from pytorchvideo.transforms import (
#     ApplyTransformToKey,
#     Normalize,
#     RandomShortSideScale,
#     RemoveKey,
#     ShortSideScale,
#     UniformTemporalSubsample,
# )

# from torchvision.transforms import (
#     Compose,
#     Lambda,
#     RandomCrop,
#     RandomHorizontalFlip,
#     Resize,
# )


model_ckpt = "MCG-NJU/videomae-base" # pre-trained model from which to fine-tune
batch_size = 8 # batch size for training and evaluation
    # "BUY":0,
    # "HOLD":1,
    # "SELL":2
def load_dataset(path):
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
                

# image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
# model = VideoMAEForVideoClassification.from_pretrained(
#     model_ckpt,
#     label2id=label2id,
#     id2label=id2label,
#     ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
# )

path = "/home/emir/Desktop/dev/datasets/ETF_RGB_Videos/labels.txt"
dataset_rooth_path = "/home/emir/Desktop/dev/datasets/ETF_RGB_Videos/"
video, labels = load_dataset(path)
# print(labels)
# label2id = {label: i for i, label in enumerate(labels)}
# id2label = {i: label for label, i in label2id.items()}
# # print(f"Unique classes: {list(label2id.keys())}.")

# model_ckpt = "MCG-NJU/videomae-base"

# image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
# model = VideoMAEForVideoClassification.from_pretrained(
#     model_ckpt,
#     label2id=label2id,
#     id2label=id2label,
#     ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
# )


vivit_cfg = VivitConfig(image_size=64)
model = VivitForVideoClassification(vivit_cfg)
for p in video:
    container = av.open(p)
    indices = sample_all_frame_indices(clip_len=container.streams.video[0].frames)
    video = read_video_pyav(container=container, indices=indices)
    print(f"video shape: {video.shape}")


metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# training_args = TrainingArguments(output_dir="./test", evaluation_strategy="epoch")

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=small_train_dataset,
#     compute_metrics=compute_metrics,
# )
