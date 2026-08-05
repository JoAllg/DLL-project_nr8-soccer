import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator
import re
from typing import Annotated, Optional, Literal
from dataclasses import asdict, is_dataclass
import os
from pathlib import Path

FieldFloat = Annotated[float, Field(ge=-1, le=1)]

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


class Area(BaseModel):
    min: tuple[FieldFloat, FieldFloat] = (-1.0, -1.0)
    max: tuple[FieldFloat, FieldFloat] = (1.0, 1.0)

    @field_validator("max")
    @classmethod
    def max_greater_eq_than_min(cls, v, info):
        min_val = info.data.get("min")
        if min_val and (v[0] < min_val[0] or v[1] < min_val[1]):
            raise ValueError("max must be greater than min in both dimensions")
        return v


class Environment(BaseModel):
    # optional fields — can be omitted
    # ---------------------------------
    n_robots_blue: int = 1
    n_robots_yellow: int = 0

    # string resolved to to <Name>OpponentPolicy options <agent> <random>
    opponent_strategy: Optional[str] = None
    opponent_model: Optional[str] = None  # if strategy =="Agent" this model is loaded

    allowed_positions_blue: Area = Area()
    allowed_positions_yellow: Area = Area()
    allowed_positions_ball: Area = Area()

    # optional fields — can be omitted
    # ---------------------------------
    # reward name -> weight pairs, resolved to _reward_{name} methods by the environment
    rewards: Optional[dict[str, float]] = None

    # reject any field not defined above
    model_config = ConfigDict(extra="forbid")


class Stage(BaseModel):
    # required fields — must be present
    # ---------------------------------
    name: str
    environment: Environment = Environment()
    total_steps: int

    # optional fields — can be omitted
    # ---------------------------------
    # rollout length; when omitted here it is filled from Config.num_steps
    steps: int = Field(default=2 * 1024, multiple_of=1024)
    n_robots_yellow: Optional[int] = None
    save_model: bool = True

    num_minibatches: Optional[int] = None

    # per-stage env-count overrides; fall back to Config.num_envs / Config.envs_per_cpu.
    # An explicit --num-envs on the CLI still wins over both.
    num_envs: Optional[int] = None
    envs_per_cpu: Optional[int] = None

    # reject any field not defined above
    model_config = ConfigDict(extra="forbid")

    @field_validator("name")
    @classmethod
    def validate_wandb_safe_name(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", v):
            raise ValueError(
                f"stage name '{v}' contains characters not allowed in wandb artifact "
                f"names (only letters, digits, '-', '_', '.' are permitted). "
            )
        return v


class Config(BaseModel):
    # required fields — must be present
    # ---------------------------------
    stages: list[Stage] = Field(min_length=1)

    # optional fields — can be omitted
    # ---------------------------------
    defaults: dict = Field(default_factory=dict)

    # reject any field not defined above
    model_config = ConfigDict(extra="forbid")

    _name_to_index: dict[str, int] = PrivateAttr(default_factory=dict)

    # command line arguments get overridden
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    # the name of this experiment"""
    seed: int = 1
    # seed of the experiment"""
    torch_deterministic: bool = False
    # if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    # if toggled, cuda will be enabled by default"""
    track: bool = False
    # if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    # the wandb's project name"""
    wandb_entity: Optional[str] = None
    # the entity (team) of wandb's project"""
    capture_video: bool = False
    # whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    # whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "SSLDynamicRobots-v0"
    # the id of the environment"""
    total_timesteps: int = 20000000
    # total timesteps of the experiments"""
    num_envs: int = 0
    # the number of parallel game environments (0 -> envs_per_cpu * available CPUs)"""
    envs_per_cpu: int = 1
    # multiplier used to derive num_envs from the available CPU count when num_envs is unset"""
    num_steps: int = 2 * 1024
    # the number of steps to run in each environment per policy rollout (see apply_num_steps)"""
    num_minibatches: int = 16  # TODO: double when running on CPU cluster
    # the number of mini-batches"""
    update_epochs: int = 4
    # the K epochs to update the policy"""
    learning_rate: float = 3e-4
    # the learning rate of the optimizer"""
    anneal_lr: bool = True
    # Toggle the cosine-with-warmup learning rate schedule for policy and value networks"""
    warmup_ratio: float = 0.01
    # fraction of total optimizer steps used for linear LR warmup at the start of each cycle (total warmup = num_cycles * this)"""
    min_lr_ratio: float = 3e-8
    # the LR floor, as a fraction of learning_rate, that the cosine schedule decays to"""
    num_cycles: int = 1
    # number of warmup+cosine-decay LR cycles across training (1 = single cycle, no restarts)"""
    cycle_decay: float = 1
    # peak-LR multiplier applied at each LR restart (0.5 halves the max LR every cycle); 1.0 = no decay"""
    weight_decay: float = 0.01
    # AdamW weight decay (applied to matrix weights only, see optimizer setup)"""
    gamma: float = 0.99
    # the discount factor gamma"""
    gae_lambda: float = 0.95
    # the lambda for the general advantage estimation"""
    norm_adv: bool = True
    # Toggles advantages normalization"""
    clip_coef: float = 0.1
    # the surrogate clipping coefficient"""
    clip_vloss: bool = True
    # Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    # initial coefficient of the entropy bonus (annealed linearly to final_ent_coef)"""
    # entropy-coefficient annealing, after cleanrl ppo_trxl.py (init/final_ent_coef):
    final_ent_coef: float = 0.0
    # final entropy coefficient after linear annealing from ent_coef over total_timesteps"""
    vf_coef: float = 0.25
    # coefficient of the value function"""
    max_grad_norm: float = 0.25
    # the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = 0.05
    # the target KL divergence threshold"""
    rpo_alpha: float = 0.2  # Best values between 0.5 to 0.1
    # the alpha parameter for RPO"""

    # Agent architecture arguments
    agent_type: Literal["mlp", "transformer"] = "transformer"
    # the actor/critic architecture: CleanRL MLP baseline or per-entity-token transformer"""
    d_model: int = 256
    # (transformer) the model/embedding dimension"""
    n_layers: int = 4
    # (transformer) the number of encoder layers"""
    n_heads: int = 8
    # (transformer) the number of attention heads (must divide d_model)"""
    ff_dim: int = 512
    # (transformer) the feedforward dimension inside encoder layers"""
    dropout: float = 0.0  # Should not be used (RPO/PPO regularize with action-mean perturbation and sampling noise)
    # (transformer) dropout inside encoder layers"""
    critic_pooling: Literal["mean", "max", "attention"] = "attention"
    # (transformer) how the critic pools entity tokens into a scalar value"""

    # to be filled in runtime
    batch_size: int = 0
    # the batch size (computed in runtime)"""
    minibatch_size: int = 0
    # the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    # the number of iterations (computed in runtime)"""
    config: str = "config.yml"
    # which stage of the config file should be executed. None: execute all stages in Order
    load_model: Optional[str] = None
    # Path to a .cleanrl_model checkpoint to load before the first training stage."""
    save_steps: int = 0
    # How often the model should be saved in between (0 -> only save at the end of a stage)"""
    stage_name: Optional[list[str]] = Field(default=None)
    rewards: Optional[str] = None
    # name of a reward_templates entry that overrides every stage's reward weights"""

    position_templates: dict[str, Area] = Field(default_factory=dict)
    # Reusable position templates (Area) for robot/ball spawn definitions"""
    reward_templates: dict[str, dict[str, float]] = Field(default_factory=dict)
    # Reusable reward-weight templates for environment.rewards"""

    # this method run after the complete model is initialized
    def model_post_init(self, __context) -> None:
        self._name_to_index = {stage.name: i for i, stage in enumerate(self.stages)}

    def apply_num_steps(self, force: bool = False) -> None:
        """Push `num_steps` into the stages' rollout length.

        A stage that sets `steps` in the yaml keeps it; `force` (an explicit
        --num-steps on the CLI) overrides every stage. Call after the CLI args
        have been merged into the config.
        """
        for stage in self.stages:
            if force or "steps" not in stage.model_fields_set:
                stage.steps = self.num_steps

    def get_stages_from_name(self, names: Optional[list[str]]) -> list[int]:
        if not names:
            return [i for i in range(len(self.stages))]
        missing = [name for name in names if name not in self._name_to_index]
        if missing:
            raise ValueError(f" stage names not found in config: {missing}")
        return [self._name_to_index[name] for name in names]


def flatten_dict(d, parent_key="", sep="-"):
    """Recursively flatten nested dicts/lists into dotted-key : value pairs."""
    items = {}
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            items.update(flatten_dict(v, new_key, sep=sep))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(flatten_dict(v, new_key, sep=sep))
    else:
        items[parent_key] = d
    return items


def override_with_args(args, config: Config) -> Config:
    """
    Override fields in a pydantic `config` with values from `args`
    (a tyro-parsed dataclass, argparse.Namespace, or dict).
    Only overrides keys that already exist as fields on `config`.
    Returns a new, validated config instance.
    """
    if is_dataclass(args):
        args_dict = asdict(args)
    elif isinstance(args, dict):
        args_dict = args
    elif hasattr(args, "__dict__"):
        args_dict = vars(args)
    else:
        raise TypeError(f"Unsupported args type: {type(args)}")

    valid_fields = config.model_fields.keys()
    updates = {k: v for k, v in args_dict.items() if k in valid_fields}

    unknown = set(args_dict.keys()) - valid_fields
    if unknown:
        print(f"Warning: ignoring args not in config: {unknown}")

    return config.model_copy(update=updates)


def load_config(path: str) -> Config:
    with open(CONFIGS_DIR / path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


# test reading the config
if __name__ == "__main__":
    config = load_config("config.yml")

    for stage in config.stages:
        print(stage.steps)
