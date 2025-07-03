import torch


def select_action_epsilon_greedy(model, state, epsilon, n_actions, device):
    """
    Sélectionne une action selon une politique epsilon-greedy.

    Args:
        model: réseau DQN
        state: torch.Tensor uint8 de forme (C, H, W) déjà sur device
        epsilon: probabilité d'exploration
        n_actions: nombre d'actions discrètes
        device: torch.device

    Returns:
        action: int
    """
    if torch.rand(1).item() < epsilon:
        # Exploration aléatoire
        return torch.randint(0, n_actions, (1,)).item()
    else:
        # Exploitation: on prédit Q et on prend argmax
        # Conversion en float et ajout batch dimension
        state_tensor = state.unsqueeze(0).float().to(device) / 255.0
        with torch.no_grad():
            q_values = model(state_tensor)
        # Retourne l'action de plus grande Q-value
        return int(q_values.argmax(dim=1)[0].item())

