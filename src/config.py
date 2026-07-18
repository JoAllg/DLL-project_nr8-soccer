import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from typing import Optional

class Environment(BaseModel):
    # optional fields — can be omitted
    # ---------------------------------
    n_robots_blue: int = 1
    n_robots_yellow: int = 0

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
    iterations: int


    # optional fields — can be omitted
    # ---------------------------------
    steps: Optional[int] = Field(default=None, multiple_of=1024)
    n_robots_yellow: Optional[int] = None
    save_model: bool = True

    # reject any field not defined above
    model_config = ConfigDict(extra="forbid")

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

    # this method run after the complete model is initialized
    def model_post_init(self, __context) -> None:
        self._name_to_index = {stage.name: i for i, stage in enumerate(self.stages)}

    def get_stages_from_name(self, names: Optional[list[str]]) -> list[int]:
        if not names:
            return [i for i in range(len(self.stages))]
        missing = [name for name in names if name not in self._name_to_index]
        if missing:
            raise ValueError(f" stage names not found in config: {missing}")
        return [self._name_to_index[name] for name in names]




def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


# test reading the config
if __name__ == "__main__":
    config = load_config("config.yml")

    for stage in config.stages:
        print(stage.steps)


