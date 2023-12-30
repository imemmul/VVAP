import numpy as np
import av
from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import Image

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


class VideoLabelDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform 

    def __len__(self):
        """
        Returns:
            int: number of rows of the csv file (not include the header).
        """
        return len(self.dataframe)

    def __getitem__(self, index):
        """ get a video and its label """
        item = {}
        video_path = self.dataframe.iloc[index].video
        label = self.dataframe.iloc[index].label
        container = av.open(video_path)
        indices = sample_all_frame_indices(clip_len=64)
        video = read_video_pyav(container=container, indices=indices)
        if self.transform:
            transformed_video = []
            for frame in video:
                frame = Image.fromarray(frame.astype(np.uint8))
                transformed_frame = self.transform(frame)
                transformed_video.append(transformed_frame)

            video = torch.stack(transformed_video)
        item['pixel_values'] = torch.FloatTensor(video)
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item