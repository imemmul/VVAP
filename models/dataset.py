import numpy as np
import av

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
    indices = list(range(clip_len))
    return indices