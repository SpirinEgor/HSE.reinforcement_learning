from dataclasses import dataclass


@dataclass
class Config:
    env_name: str = "Walker2DBulletEnv-v0"

    lambda_: float = 0.97
    gamma: float = 0.99

    actor_lr: float = 3e-4
    actor_hidden_dim: int = 128
    actor_hidden_layers: int = 1

    critic_lr: float = 2e-4
    critic_hidden_dim: int = 128
    critic_hidden_layers: int = 1

    clip: float = 0.2
    entropy_cf: float = 1e-2
    batches_per_update: int = 64
    batch_size: int = 64

    min_transitions_per_update: int = 2048
    min_episodes_per_update: int = 4

    iterations: int = 1000

    seed: int = 7

    eval_episodes: int = 50
