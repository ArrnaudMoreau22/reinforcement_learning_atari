
import os
import time
import random
import gc
import gymnasium as gym
# Importation de ale_py pour les environnements Atari obligatoires
# Note: ale_py est une dépendance de gymnasium pour les environnements Atari
import ale_py
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gymnasium.wrappers import RecordVideo, AtariPreprocessing
from preprocessing import FrameStack
from replay_buffer import ReplayBuffer

from model import DQN
from train_step import train_step
from agent_utils import select_action_epsilon_greedy



class NoopResetEnv(gym.Wrapper):
    """
    Wrapper pour injecter 1–noop_max actions NOOP aléatoires après chaque reset.
    """
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = random.randint(1, self.noop_max)
        for _ in range(noops):
            obs, reward, terminated, truncated, info = self.env.step(self.noop_action)
            done = terminated or truncated
            if done:
                obs, info = self.env.reset(**kwargs)
        return obs, info

class FireResetEnv(gym.Wrapper):
    """
    Envoie l'action FIRE automatiquement après chaque reset.
    Utile pour les jeux comme Breakout où il faut appuyer sur FIRE pour commencer.
    """
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        self.fire_action = 1  # action "FIRE"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(self.fire_action)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info

class FireOnLifeLossEnv(gym.Wrapper):
    """
    Force l'action FIRE (action 1) après chaque perte de vie, y compris après reset().
    Utilisable uniquement avec des environnements de type Atari.
    """
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True  # pour différencier reset vs perte de vie
        self.fire_action = 1  # l'action FIRE est 1 dans Breakout

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.lives = info.get("lives", 0)
        # FIRE après reset, obligatoire dans Breakout
        obs, _, terminated, truncated, info = self.env.step(self.fire_action)
        self.was_real_done = True
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        current_lives = info.get("lives", 0)

        if current_lives < self.lives and current_lives > 0:
            # Vie perdue mais pas fin de jeu → injecter FIRE
            obs, _, terminated, truncated, info = self.env.step(self.fire_action)
            done = terminated or truncated

        self.was_real_done = done
        self.lives = current_lives
        return obs, reward, terminated, truncated, info

def train_dqn(
    env_id: str,
    total_steps: int,
    buffer_capacity: int,
    batch_size: int,
    gamma: float,
    train_freq: int,
    target_update_steps: int,
    lr: float,
    alpha: float,
    INITIAL_REPLAY_SIZE: int,
    video_interval: int,
    plot_interval: int,
    save_path: str,
    resume_ckpt: str = None,
):
    """
    Entraîne un agent DQN sur Breakout-v5 (Gymnasium), avec :
    - reward clipping corrigé (clip après avoir sommé les 4 frames),
    - target network update tous les target_update_steps (= nombre d'updates),
    - burn-in initial (50k transitions) même au premier lancement.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # 1) Pré-allocation du replay buffer
    replay = ReplayBuffer(capacity=buffer_capacity, device='cpu')

    # 2) Création dossiers vidéo + plots
    timestamp = int(time.time())
    record_dir = f"videos/run_{timestamp}"
    os.makedirs(record_dir, exist_ok=True)
    plots_dir = os.path.join(record_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 3) Préparation de la figure métriques
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax_score, ax_eps, ax_loss, ax_q = axs.flatten()
    fig.suptitle("DQN – Breakout")

    # 4) Environnement Atari + No-op reset + pas de terminaison sur perte de vie
    base_env = gym.make(env_id, render_mode="rgb_array", frameskip=1)
    base_env = NoopResetEnv(base_env, noop_max=30)
    # Note: FireResetEnv et FireOnLifeLossEnv sont optionnels selon le jeu (actuellement sur pong / mais nécessaire pour Breakout)
    # base_env = FireResetEnv(base_env)
    # base_env = FireOnLifeLossEnv(base_env)

    env = AtariPreprocessing(
        base_env,
        frame_skip=4,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
        screen_size=84
    )
    env.reset(seed=SEED)
    env.action_space.seed(SEED)

    # 5) Instanciation des réseaux
    n_actions = env.action_space.n
    policy_net = DQN(input_channels=4, num_actions=n_actions).to(device)
    target_net = DQN(input_channels=4, num_actions=n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = torch.optim.RMSprop(
        policy_net.parameters(), lr=lr, alpha=alpha, eps=0.01
    )

    # 6) Variables de monitoring / reprise
    step_idx    = 0     # nombre de steps effectuées
    train_steps  = 0     # nombre d'updates (train_step appelés)
    episode      = 0
    scores, epsilons, losses, q_means, step_history = [], [], [], [], []


    # 7) Si on reprend un checkpoint, on re-charge modèle + opti + compteurs
    if resume_ckpt is not None:
        ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        policy_net.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step_idx = ckpt["step_idx"]
        episode   = ckpt["episode"]
        print(f"▶ Reprise du modèle à step {step_idx}, épisode {episode}")

    # 8) **Burn-in initial** (random) jusqu’à INITIAL_REPLAY_SIZE, même au tout premier run
    print(f"▶ Burn-in initial du buffer jusqu'à {INITIAL_REPLAY_SIZE} transitions…")
    obs, _ = env.reset()
    frame_stack = FrameStack(4, device=device)
    state = frame_stack.reset(obs)
    for _ in range(INITIAL_REPLAY_SIZE):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        clipped_r = np.sign(reward)
        next_state = frame_stack.step(next_obs)
        done = terminated
        replay.push(state.cpu().numpy().astype(np.uint8), action, clipped_r, next_state.cpu().numpy().astype(np.uint8), done)
        state = next_state if not done else frame_stack.reset(env.reset()[0])
    
    print("▶ Burn-in initial terminé. Début de l'entraînement effectif.")
    

    # 9) Enregistrement vidéo des épisodes tous les `video_interval` épisodes
    env = RecordVideo(
        env,
        video_folder=record_dir,
        episode_trigger=lambda ep: (ep + 1) % video_interval == 0,
        name_prefix="ep"
    )

    # 10) Initialisation finale du frame stack avant la boucle principale
    obs, _ = env.reset()
    frame_stack = FrameStack(4, device=device)
    state = frame_stack.reset(obs)

    # 11) Fonction ε-greedy identique à l'article DQN vanilla
    def epsilon_by_steps(s):
        if s < 1_000_000:
        # Linéaire de 1.0 à 0.01 sur 1M steps
            return 1.0 - 0.99 * (s / 1_000_000)
        else:
            # Fixé à 0.01 ensuite
            return 0.01

    try:
        # ------- Boucle principale -------
        while step_idx < total_steps:
            obs, _ = env.reset()
            state = frame_stack.reset(obs)
            done = False
            ep_reward = 0.0
            ep_losses = []
            episode += 1
            gc.collect()

            while not done and step_idx < total_steps:
                eps = epsilon_by_steps(step_idx)
                # 11.1) Choix d'action ε-greedy
                action = select_action_epsilon_greedy(policy_net, state, eps, n_actions, device)

                # 11.2) Frame-skip + sum raw rewards
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated
                step_idx += 1

                # 11.3) Clip unique de la somme sur les 4 frames
                clipped_r = np.sign(reward)

                # 11.4) Empiler la nouvelle observation
                next_state = frame_stack.step(next_obs)
                replay.push(
                    state.cpu().numpy().astype(np.uint8),
                    action,
                    clipped_r,
                    next_state.cpu().numpy().astype(np.uint8),
                    done
                )
                state = next_state
                ep_reward += reward

                if step_idx % train_freq == 0 and len(replay) >= INITIAL_REPLAY_SIZE:
                    loss = train_step(
                        policy_net, target_net, optimizer,
                        replay, batch_size, gamma, device)
                    train_steps += 1
                    ep_losses.append(loss)

                    with torch.no_grad():
                        sb, *_ = replay.sample(batch_size)
                        sb = sb.float().to(device) / 255.0
                        q_means.append(policy_net(sb).max(1)[0].mean().item())
                        step_history.append(step_idx)

                    # Mise à jour du target network
                    if train_steps % target_update_steps == 0:
                        target_net.load_state_dict(policy_net.state_dict())

            # ------- Fin d'épisode -------
            scores.append(ep_reward)
            epsilons.append(eps)
            losses.append(np.mean(ep_losses) if ep_losses else 0.0)
            print(f"Ep {episode} | Step {step_idx} | Score {ep_reward:.1f} | ε {eps:.3f} | Loss {losses[-1]:.4f}")


            # 11.7) Sauvegarde des métriques tous les plot_interval épisodes
            if episode % plot_interval == 0:
                # Score + MA100
                ax_score.clear()
                ax_score.scatter(range(1, len(scores)+1), scores, alpha=0.3)
                if len(scores) >= 100:
                    ma = np.convolve(scores, np.ones(100)/100, mode='valid')
                    ax_score.plot(range(100, len(scores)+1), ma, 'r')
                ax_score.set(title='Score (bricks)', xlabel='Episode', ylabel='Score')

                # ε
                ax_eps.clear()
                ax_eps.plot(range(1, len(epsilons)+1), epsilons)
                ax_eps.set(title='ε', xlabel='Episode', ylabel='ε')

                # Loss MA500
                ax_loss.clear()
                if losses:
                    ma500 = np.convolve(losses, np.ones(500)/500, mode='valid')
                    ax_loss.plot(np.arange(len(ma500)), ma500)
                ax_loss.set(title='Loss (MA500)', xlabel='Itérations de train', ylabel='MSE')

                # Q_mean
                ax_q.clear()
                ax_q.plot(step_history, q_means)
                ax_q.set(title='Q_mean', xlabel='Steps', ylabel='Q')

                fig.tight_layout(rect=[0, 0, 1, 0.96])
                fig.savefig(os.path.join(plots_dir, f"metrics_ep{episode}.png"))

    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user, saving checkpoint...")

    finally:
        # 12) Sauvegarde du checkpoint
        ckpt = {
            'model_state_dict':     policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step_idx':            step_idx,
            'episode':              episode,
            'train_steps':          train_steps,
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(ckpt, save_path, _use_new_zipfile_serialization=False, pickle_protocol=4)
        print(f"✔ Checkpoint saved to {save_path}")
        env.close()
 