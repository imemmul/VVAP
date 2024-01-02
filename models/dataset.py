import numpy as np
import av
import random
from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import Image

def custom_collate_fn(batch):
    '''
    this collate is not a cool choice btw
    '''
    pixel_values_list = []
    labels_list = []

    max_length = max([item['pixel_values'].shape[0] for item in batch])

    for item in batch:
        pixel_values = item['pixel_values']
        label = item['labels']

        padding_size = max_length - pixel_values.shape[0]

        if padding_size > 0:
            padding = torch.zeros((padding_size, *pixel_values.shape[1:]), dtype=pixel_values.dtype)
            pixel_values = torch.cat((pixel_values, padding), dim=0)

        pixel_values_list.append(pixel_values)
        labels_list.append(label)

    pixel_values_batch = torch.stack(pixel_values_list)
    labels_batch = torch.stack(labels_list)

    return {'pixel_values': pixel_values_batch, 'labels': labels_batch}



def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_all_frame_indices(clip_len):
    '''
    Generate all frame indices for a video with a fixed number of frames.
    Args:
        clip_len (`int`): Total number of frames in the video.
    Returns:
        indices (`List[int]`): List of all frame indices.
    '''
    indices = np.array(range(clip_len))
    return indices

import ast
class VideoLabelDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.dataframe['labels'] = self.dataframe['labels'].apply(self.parse_labels)
        
        self.transform = transform
        self.max_clip_len = 64

    def __len__(self):
        """
        Returns:
            int: number of rows of the csv file (not include the header).
        """
        return len(self.dataframe)
    def parse_labels(self, label_str):
        '''
        convert string cell into correct format
        '''
        try:
            return ast.literal_eval(label_str)
        except ValueError:
            return []  # or some default value like [0, 0, 0]
        
    # def __getitem__(self, index):
    #     """ get a video and its label """
    #     item = {}
    #     video_path = self.dataframe.iloc[index].video
    #     label = self.dataframe.iloc[index].label
    #     container = av.open(video_path)
    #     indices = sample_all_frame_indices(clip_len=64)
    #     video = read_video_pyav(container=container, indices=indices)
    #     if self.transform:
    #         transformed_video = []
    #         for frame in video:
    #             frame = Image.fromarray(frame.astype(np.uint8))
    #             transformed_frame = self.transform(frame)
    #             transformed_video.append(transformed_frame)

    #         video = torch.stack(transformed_video)
    #     item['pixel_values'] = torch.FloatTensor(video)
    #     item['labels'] = torch.tensor(label, dtype=torch.long)
    #     return item
    def __getitem__(self, index):
        video_path = self.dataframe.iloc[index].file_name
        label = self.dataframe.iloc[index].labels
        container = av.open(video_path)
        
        indices = sample_all_frame_indices(clip_len=self.max_clip_len)

        # temporal cropping
        start = random.randint(0, max(0, len(indices) - self.max_clip_len))
        end = start + self.max_clip_len
        indices = indices[start:end]

        video = read_video_pyav(container=container, indices=indices)

        if self.transform:
            video = [self.transform(Image.fromarray(frame.astype(np.uint8))) for frame in video]
            video = torch.stack(video)

        return {'pixel_values': video, 'labels': torch.tensor(label, dtype=torch.long)}