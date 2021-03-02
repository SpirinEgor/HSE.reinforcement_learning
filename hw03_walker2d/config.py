from dataclasses import dataclass


@dataclass
class Config:
    env_name = "Walker2DBulletEnv-v0"

    lambda_ = 0.95
    gamma = 0.99

    actor_lr = 2e-4
    actor_hidden_dim = 1024
    actor_hidden_layers = 5

    critic_lr = 1e-4
    critic_hidden_dim = 1024
    critic_hidden_layers = 5

    clip = 0.2
    entropy_cf = 1e-2
    batches_per_update = 64
    batch_size = 64

    min_transitions_per_update = 2048
    min_episodes_per_update = 4

    iterations = 1000

    seed = 7

    eval_episodes = 50
