import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

# Singleton scaler for mixed precision
scaler = GradScaler(device='cuda')

def train_step(
    model,
    target_model,
    optimizer,
    replay_buffer,
    batch_size,
    gamma,
    device
):
    """
    Effectue une itération de mise à jour pour le DQN vanilla.

    Args:
        model: réseau principal (policy network)
        target_model: réseau cible (target network)
        optimizer: optimiseur RMSProp
        replay_buffer: ReplayBuffer à échantillonnage uniforme
        batch_size: taille du minibatch
        gamma: facteur d'actualisation
        device: torch.device ('cpu' ou 'cuda')

    Returns:
        loss.item()
    """
    model.train()

    # 1) Échantillonnage
    ## -- Replay Buffer --
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)

    # Passage en float32 + normalisation (0–1) pour le modèle
    ## Version double DQN
    # state_batch      = state_batch.to(device).float() / 255.0
    # next_state_batch = next_state_batch.to(device).float() / 255.0
    # action_batch     = action_batch.to(device)
    # reward_batch     = reward_batch.to(device)
    # done_batch       = done_batch.to(device)

    states      = state_batch.to(device).float() / 255.0
    next_states = next_state_batch.to(device).float() / 255.0
    actions     = action_batch.to(device)
    rewards     = reward_batch.to(device)
    dones       = done_batch.to(device)

    # 2) Calcul des Q(s,a)
    # On récupère les Q values pour chacune des states du batch.
    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)


    # 3) Calcul des cibles TD
    ## -- DQN vanilla --
    # with torch.no_grad():
    #     # On récupère les Q values pour les next_states du batch via le target_model.
    #     max_next_q = target_model(next_states).max(dim=1)[0]
    #     # On calcule les cibles TD qui sont les récompenses + gamma * max_next_q qui servent à estimer la valeur future.
    #     # Reward est normalisé et correspond à la somme des récompenses cumulées.
    #     # Gamma est le facteur d'actualisation (discount factor).
    #     # Max next Q est la valeur maximale des Q values pour les next_states.
    #     target_q = rewards + gamma * max_next_q * (1 - dones)
        # target_q est un vecteur de taille batch_size, contenant les cibles TD pour chaque état du batch.


    ## -- Double DQN --
    with torch.no_grad():
        next_actions = model(next_states).argmax(dim=1)
        next_q = target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q = rewards + gamma * next_q * (1 - dones)

    # La MSE Loss est calculée entre les Q values prédites et les cibles TD.
    # q_values est un vecteur de taille batch_size, contenant les Q values prédites pour chaque état du batch.
    # target_q est un vecteur de taille batch_size, contenant les cibles TD pour chaque état du batch.
    # La loss est la moyenne des erreurs au carré entre les Q values prédites et les cibles TD.
    #Au choix l'une ou l'autre mais la huber loss est plus robuste aux grands changements de Q values.
    # loss = F.mse_loss(q_values, target_q) 
    loss = F.huber_loss(q_values, target_q)


    # 5) Backpropagation + gradient clipping
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    return loss.item()