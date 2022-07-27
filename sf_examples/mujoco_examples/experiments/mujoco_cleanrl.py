from sample_factory.runner.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("seed", [0, 1111, 2222, 3333, 4444]),
        (
            "env-id",
            [
                "Ant-v4",
                "HalfCheetah-v4",
                "Hopper-v4",
                "Humanoid-v4",
                "InvertedDoublePendulum-v4",
                "InvertedPendulum-v4",
                "Reacher-v4",
                "Swimmer-v4",
                "Walker2d-v4",
            ],
        ),
    ]
)

_experiments = [
    Experiment(
        "mujoco_cleanrl",
        "python -m sf_examples.mujoco_examples.cleanrl --total-timesteps 10000000 --num-envs 64 --anneal-lr=True --clip-vloss=False --ent-coef=0 --learning-rate=0.00295 --max-grad-norm=3.5 --num-minibatches=4 --num-steps=64 --update-epochs=2 --vf-coef=1.3 --track --wandb-project-name mujoco-cleanrl",
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription(
    "mujoco_cleanrl", experiments=_experiments, experiment_arg_name="--exp-name", experiment_dir_arg_name=None
)
# python -m sample_factory.runner.run --run=sf_examples.mujoco_examples.experiments.mujoco_cleanrl --runner=processes --max_parallel=1  --pause_between=1 --experiments_per_gpu=10000 --num_gpus=1
