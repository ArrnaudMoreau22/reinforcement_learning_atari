from trainer import train_dqn
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",             type=str,   default="ALE/Pong-v5")
    parser.add_argument("--total-steps",     type=int,   default=50_000_000)
    parser.add_argument("--buffer-cap",      type=int,   default=500_000)
    parser.add_argument("--batch-size",      type=int,   default=32)
    parser.add_argument("--gamma",           type=float, default=0.99)
    parser.add_argument("--train-freq",      type=int,   default=4, help="Fréquence (en steps) des mises à jour du réseau")
    parser.add_argument("--target-update",   type=int,   default=10000, help="Fréquence (en train_steps) de la synchro du target_net")
    parser.add_argument("--lr",              type=float, default=0.00025)
    parser.add_argument("--alpha",           type=float, default=0.95)
    parser.add_argument("--initial-replay",  type=int,   default=50000)
    parser.add_argument("--video-interval",  type=int,   default=2000)
    parser.add_argument("--plot-interval",   type=int,   default=2000)
    parser.add_argument("--resume",          type=str,   default=None)
    parser.add_argument("--save-path",       type=str,   default="checkpoints/dqn_breakout.pth")
    args = parser.parse_args()

    train_dqn(
        env_id=args.env,
        total_steps=args.total_steps,
        buffer_capacity=args.buffer_cap,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=args.train_freq,
        target_update_steps=args.target_update,
        lr=args.lr,
        alpha=args.alpha,
        INITIAL_REPLAY_SIZE=args.initial_replay,
        video_interval=args.video_interval,
        plot_interval=args.plot_interval,
        save_path=args.save_path,
        resume_ckpt=args.resume
    )
