import random
import numpy as np
import torch

class ReplayBuffer:
    """
    Buffer de rejouabilité classique, sans priorité.
    Chaque transition est stockée sous forme de tableau NumPy uint8 pour réduire l'utilisation mémoire.
    Les données sont converties en tenseurs Torch lors de l'échantillonnage (sample).
    """
    def __init__(self, capacity, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """
        Stocke une transition dans le buffer. Écrase les plus anciennes une fois la capacité atteinte.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max(self.priorities, default=1.0))
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = max(self.priorities, default=1.0)

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Échantillonne un batch de transitions aléatoires et les retourne sous forme de tenseurs.
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))

        # Conversion en tenseurs
        state      = torch.tensor(state, dtype=torch.uint8, device=self.device)  # (B, 4, 84, 84)
        action     = torch.tensor(action, dtype=torch.int64, device=self.device) # (B,)
        reward     = torch.tensor(reward, dtype=torch.float32, device=self.device) # (B,)
        next_state = torch.tensor(next_state, dtype=torch.uint8, device=self.device) # (B, 4, 84, 84)
        done       = torch.tensor(done, dtype=torch.float32, device=self.device) # (B,)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
