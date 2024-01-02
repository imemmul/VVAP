import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
import os
import pandas as pd
import cv2
import json

# print(loaded_data.shape)

### tsf indicator deleted to get 64x64

etfList = ['XLF', 'XLU', 'QQQ', 'SPY', 'XLP', 'EWZ', 'EWH', 'XLY', 'XLE']


indicators_str = [
    'rsi', 'cmo', 'plus_di', 'minus_di', 'willr', 'cci', 'ultosc', 'aroonosc', 'mfi', 'mom', 'macd', 'macdfix',
    'linearreg_angle', 'linearreg_slope', 'rocp', 'roc', 'rocr', 'rocr100', 'slowk', 'fastd', 'slowd', 'aroonup',
    'aroondown', 'apo', 'macdext', 'fastk', 'ppo', 'minus_dm', 'adosc', 'fastdrsi', 'fastkrsi', 'trange', 'trix',
    'std', 'bop', 'var', 'plus_dm', 'correl', 'ad', 'beta', 'wclprice', 'typprice', 'avgprice', 'medprice',
    'bbands_lowerband', 'linearreg', 'obv', 'bbands_middleband', 'tema', 'bbands_upperband', 'dema', 'midprice',
    'midpoint', 'wma', 'ema', 'ht_trendline', 'kama', 'sma', 'ma', 'adxr', 'adx', 'trima', 'linearreg_intercept', 'dx'
    ]


def createRGBVideo(rgb_frames, rgb_labels, dataset_dir, etf_names, group_idx, frame_rate=5, frames_per_video=64):
    video_info = {}
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    group_dir = os.path.join(dataset_dir, f"group_{group_idx}")
    os.makedirs(group_dir, exist_ok=True)

    num_videos = len(rgb_frames) - (frames_per_video - 1)
    for i in range(num_videos):
        start_idx = i
        end_idx = start_idx + frames_per_video

        # Create a label vector for each video
        video_labels = rgb_labels[end_idx - 1, :].tolist()
        # label_name = "_".join([str(int(label)) for label in video_labels])

        # Construct the video filename
        video_filename = os.path.join(group_dir, f'output_video_{i+1}.mp4')

        # Write the video frames to a file
        with imageio.get_writer(video_filename, fps=frame_rate, format='mp4', codec='libx264') as writer:
            for frame in rgb_frames[start_idx:end_idx]:
                frame = (frame * 255).astype(np.uint8)
                writer.append_data(frame)

        video_info[video_filename] = video_labels
        if end_idx == len(rgb_frames):
            break
    print(video_info)
    return video_info

import torch
import torch.nn.functional as F
def main(x_etfs, y_etfs, etf_groups, test):
    rgb_frames = []
    rgb_labels = []
    video_info = {}
    dataset_dir = "/home/emir/Desktop/dev/datasets/ETF_RGB_Videos" if not test else "/home/emir/Desktop/dev/datasets/ETF_RGB_Videos_Test"
    print(etf_groups)

    for i in range(len(x_etfs)):
        frame = x_etfs[i].transpose((1, 2, 3, 0))
        print(frame.shape)

        label = y_etfs[i].squeeze().transpose(1, 0)
        # print(label.shape)
        # rgb_labels.append(label)
        label_tensor = torch.tensor(y_etfs[i].squeeze().transpose(1, 0), dtype=torch.int64)
        # one_hot_labels = F.one_hot(label_tensor, num_classes=2)
        # print(one_hot_labels)
        rgb_labels.append(label_tensor)

        rgb_frames.append(frame)

    for idx, (etf_group, f, labels) in enumerate(zip(etf_groups, rgb_frames, rgb_labels)):
        info = createRGBVideo(rgb_frames=np.array(f), dataset_dir=dataset_dir, etf_names=etf_group, group_idx=idx, rgb_labels=np.array(labels))
        video_info = {**video_info, **info}
    video_df = pd.DataFrame(video_info.items(), columns=['file_name', 'labels'])

    csv_filename = os.path.join(dataset_dir, "labels.csv")
    video_df.to_csv(csv_filename, index=False)
                
        

if __name__ == "__main__":
    # train_data = "/home/emir/Desktop/dev/datasets/ETF_new/TrainData/"
    test_data = "/home/emir/Desktop/dev/datasets/ETF_new/TestData/"
    x_loaded_etfs = []
    y_loaded_etfs = []
    etf_groups = []
    
    for i in range(0, len(etfList), 3):
        temp_l_x = []
        temp_l_y = []
        etf_groups.append(tuple(etfList[i:i+3]))
        for etf in etfList[i:i+3]:
            temp_l_x.append(np.load(os.path.join(test_data, f"x_{etf}.npy")))
            temp_l_y.append(np.load(os.path.join(test_data, f"y_{etf}.npy")))
        x_loaded_etfs.append(temp_l_x)
        y_loaded_etfs.append(temp_l_y)
    main(np.array(x_loaded_etfs), np.array(y_loaded_etfs), etf_groups, test=True)
         
