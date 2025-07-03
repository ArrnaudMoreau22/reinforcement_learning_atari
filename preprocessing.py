import torch
from collections import deque

class FrameStack:
    def __init__(self, k, device="cpu"):
        self.k      = k
        self.frames = deque([], maxlen=k)
        self.device = device

    def reset(self, obs):
        # obs : déjà (84,84) uint8 grayscale par AtariPreprocessing
        frame = self._to_tensor(obs)
        for _ in range(self.k):
            self.frames.append(frame)
        return torch.cat(list(self.frames), dim=0).to(self.device)

    def step(self, obs):
        frame = self._to_tensor(obs)
        self.frames.append(frame)
        return torch.cat(list(self.frames), dim=0).to(self.device)

    def _to_tensor(self, obs):
        # obs est un array numpy uint8 shape (84,84) ou (84,84,1)
        frame = torch.from_numpy(obs)            # -> uint8 tensor
        if frame.ndim == 2:
            frame = frame.unsqueeze(0)           # -> (1,84,84)
        return frame
