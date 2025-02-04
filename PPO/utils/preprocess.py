import cv2
import numpy as np
from collections import deque


def preprocess_frame(frame, low, high, new):
    """
    Preprocess a single frame:
      1. Crop the score area
      2. Convert to grayscale
      3. Resize to new x new
      4. Normalize [0, 255] -> [0, 1]

    :param frame:  A raw frame in RGB order.
    :return:       A preprocessed grayscale frame of shape (new, new).
    """
    # 1) Crop (remove top and bottom borders)
    cropped_frame = frame[low:high, :, :]

    # 2) Convert to grayscale
    gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2GRAY)

    # 3) Resize to 84x84
    resized_frame = cv2.resize(gray_frame, (new, new), interpolation=cv2.INTER_AREA)

    # 4) Normalize pixel values
    normalized_frame = resized_frame.astype(np.float32) / 255.0

    return normalized_frame


class FrameStack:
    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.frames = deque([], maxlen=stack_size)

    def reset(self, first_frame):
        # Clear and fill with the initial frame
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(first_frame)
        return self._get_stacked_frames()

    def add_frame(self, new_frame):
        # Append the new frame, automatically discarding the oldest
        self.frames.append(new_frame)
        return self._get_stacked_frames()

    def _get_stacked_frames(self):
        # Stack along the last axis: shape (84, 84, stack_size)
        return np.stack(self.frames, axis=-1)