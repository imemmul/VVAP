import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
import os
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

def createRGBVideo(rgb_frames, rgb_labels, dataset_dir, etf_name, group_idx, top_etf, frame_rate=5, num_videos=100):
    
    video_info = {}
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
    # for group_idx, (group_frames, group_labels) in enumerate(zip(rgb_frames, rgb_labels)):
    frames_per_video = len(rgb_frames) // num_videos
    
    group_dir = os.path.join(dataset_dir, f"group_{group_idx}")
    if not os.path.exists(group_dir):
        os.makedirs(group_dir, exist_ok=True)
    if not os.path.exists(os.path.join(group_dir, etf_name)):
        os.makedirs(os.path.join(group_dir, etf_name), exist_ok=True)
    for i in range(num_videos):
        start_idx = i * frames_per_video
        end_idx = (i + 1) * frames_per_video if i != num_videos - 1 else len(rgb_frames)

        video_label = rgb_labels[end_idx - 1, :].tolist()[top_etf]
        video_filename = os.path.join(group_dir, etf_name, f'output_video_{i+1}.mp4')

        with imageio.get_writer(video_filename, fps=frame_rate, format='mp4', codec='libx264') as writer:
            for frame in rgb_frames[start_idx:end_idx]:
                frame = (frame * 255).astype(np.uint8)
                writer.append_data(frame)

        video_info[video_filename] = video_label

    return video_info

def main(x_etfs, y_etfs, etf_groups):
    rgb_frames = []
    rgb_labels = []
    video_info = {}
    dataset_dir = "/home/emir/Desktop/dev/datasets/ETF_RGB_Videos/"
    print(etf_groups)
    for i in range(len(x_etfs)):
        frame = x_etfs[i].transpose((1, 2, 3, 0))
        print(frame.shape)

        label = y_etfs[i].squeeze().transpose(1, 0)
        print(label.shape)
        rgb_labels.append(label)

        red_replaced_by_green = np.copy(frame)
        red_replaced_by_blue = np.copy(frame)
        red_replaced_by_green[:, :, :, 0] = frame[:, :, :, 1]  # Red replaced by Green
        red_replaced_by_blue[:, :, :, 0] = frame[:, :, :, 2]   # Red replaced by Blue
        rgb_frames.append((frame, red_replaced_by_green, red_replaced_by_blue))
    
    for idx, ((etf1, etf2, etf3), (f, s, t), labels) in enumerate(zip(etf_groups, rgb_frames, rgb_labels)):
        # print(idx)
        # print(etf1, etf2, etf3)
        # print(f.shape)
        # print(s.shape)

        info_1 = createRGBVideo(rgb_frames=np.array(f), dataset_dir="/home/emir/Desktop/dev/datasets/ETF_RGB_Videos/", etf_name=etf1, group_idx=idx, rgb_labels=np.array(labels), top_etf=0)
        info_2 = createRGBVideo(rgb_frames=np.array(s), dataset_dir="/home/emir/Desktop/dev/datasets/ETF_RGB_Videos/", etf_name=etf2, group_idx=idx, rgb_labels=np.array(labels), top_etf=1)
        info_3 = createRGBVideo(rgb_frames=np.array(t), dataset_dir="/home/emir/Desktop/dev/datasets/ETF_RGB_Videos/", etf_name=etf3, group_idx=idx, rgb_labels=np.array(labels), top_etf=2)
        video_info = {**video_info, **info_1, **info_2, **info_3}
    with open(os.path.join(dataset_dir, "labels.txt"), 'w') as file:
        for key, value in video_info.items():
            if value == 0.0:
                file.write(f'{key}\tBUY\n')
            elif value == 1.0:
                file.write(f'{key}\tHOLD\n')
            elif value == 2.0:
                file.write(f'{key}\tSELL\n')
                
        

if __name__ == "__main__":
    train_data = "/home/emir/Desktop/dev/datasets/ETF/rectangle/01/TrainData"
    x_loaded_etfs = []
    y_loaded_etfs = []
    etf_groups = []
    
    for i in range(0, len(etfList), 3):
        temp_l_x = []
        temp_l_y = []
        etf_groups.append(tuple(etfList[i:i+3]))
        for etf in etfList[i:i+3]:
            temp_l_x.append(np.load(os.path.join(train_data, f"x_{etf}.npy")))
            temp_l_y.append(np.load(os.path.join(train_data, f"y_{etf}.npy")))
        x_loaded_etfs.append(temp_l_x)
        y_loaded_etfs.append(temp_l_y)
    main(np.array(x_loaded_etfs), np.array(y_loaded_etfs), etf_groups)
         
