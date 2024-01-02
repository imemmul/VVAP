import numpy as np
from transformers import VivitForVideoClassification, VivitImageProcessor, VivitModel, VivitConfig
import av
import torch
from dataset import sample_all_frame_indices, read_video_pyav
import pandas as pd
np.random.seed(0)

model = VivitForVideoClassification.from_pretrained("/home/emir/Desktop/dev/VVAP/models/model_output/checkpoint-500").to("cuda")
image_processor = VivitImageProcessor(do_resize=False, crop_size=64, do_normalize=False, offset=False, do_center_crop=False)
correct = 0
df = pd.read_csv("/home/emir/Desktop/dev/datasets/ETF_RGB_Videos_Test/dataset.csv")
# df = pd.read_csv("/home/emir/Desktop/dev/datasets/ETF_RGB_Videos/dataset.csv")
total = len(df)
# print(image_processor)

for i in range(1000):
    print(i)
    test_video_path, test_label = df['video'].iloc[i], df['label'].iloc[i]
    container = av.open(test_video_path)
    indices = sample_all_frame_indices(64)
    video = read_video_pyav(container, indices)
    # Debugging: Check the shape of video before processing
    # print("Video shape before processing:", video.shape)
    video = np.transpose(video, (0, 3, 1, 2))
    inputs = image_processor(video, return_tensors="pt").to("cuda")
    inputs['pixel_values'] = inputs['pixel_values'].squeeze().unsqueeze(0).float()
    # print("Inputs shape:", inputs['pixel_values'].shape)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_label = logits.argmax(-1).item()
    # print(model.config.id2label[predicted_label])
    # print(f"real_label: {test_label}")
    if int(predicted_label) == int(test_label):
        correct += 1 
print(f"Test Accuracy: {correct/total}")
