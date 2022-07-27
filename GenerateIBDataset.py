import ray
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.agents.trainer_template import build_trainer

RandomTrainer = build_trainer(
    name="RandomTrainer",
    default_policy=RandomPolicy
)

config = {"env": "CartPole-v1", "num_gpus": 0, "num_workers": 1, "framework": None}
ray.init(ignore_reinit_error=True, local_mode=True)
ray.tune.run(
    RandomTrainer,
    config=config,
    stop={"timesteps_total": 5},
)
ray.shutdown()