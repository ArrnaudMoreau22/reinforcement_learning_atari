import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_channels: int, num_actions: int):
        super(DQN, self).__init__()
        # Trois couches convolutionnelles pour extraire les caractéristiques spatiales
        # 1) Conv1: 32 filtres 8x8, stride 4 --> sortie (batch, 32, 20, 20)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32,
                               kernel_size=8, stride=4)
        # 2) Conv2: 64 filtres 4x4, stride 2 --> sortie (batch, 64, 9, 9)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=4, stride=2)
        # 3) Conv3: 64 filtres 3x3, stride 1 --> sortie (batch, 64, 7, 7)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=1)

        # Calcul de la taille du tenseur après les convolutions
        # 84x84 --> conv1 --> 20x20 --> conv2 --> 9x9 --> conv3 --> 7x7
        conv_output_size = 64 * 7 * 7

        # Couches entièrement connectées
        # 3136 entrées (64*7*7) --> 512 neurones cachés
        self.fc1 = nn.Linear(in_features=conv_output_size, out_features=512)
        # 512 neurones --> num_actions sorties (Q-values)

        # Normal DQN: une seule sortie
        # self.fc2 = nn.Linear(in_features=512, out_features=num_actions)

        # Dueling DQN: deux sorties
        self.fc_adv = nn.Linear(512, num_actions)  # Avantage A(s,a)
        self.fc_val = nn.Linear(512, 1)            # Valeur V(s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, input_channels=4, 84, 84)
        x = F.relu(self.conv1(x))  # --> (batch, 32, 20, 20)
        x = F.relu(self.conv2(x))  # --> (batch, 64, 9, 9)
        x = F.relu(self.conv3(x))  # --> (batch, 64, 7, 7)

        # Aplatir pour la partie fully connected
        x = x.view(x.size(0), -1)  # --> (batch, 64*7*7 = 3136)
        x = F.relu(self.fc1(x))    # --> (batch, 512)

        # Dueling DQN: calcul des Q-values
        adv = self.fc_adv(x)        # → (batch, num_actions)
        val = self.fc_val(x)        # → (batch, 1)
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q

        # Couche de sortie linéaire donnant les Q-values pour chaque action
        # Normal DQN
        # return self.fc2(x)         # --> (batch, num_actions)
