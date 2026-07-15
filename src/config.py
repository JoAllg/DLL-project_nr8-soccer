import yaml
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional

class Environment(BaseModel):
    # optional fields — can be omitted
    # ---------------------------------
    n_robots_blue: int = 1
    n_robots_yellow: int = 0

    # reject any field not defined above
    model_config = ConfigDict(extra="forbid")

class Stage(BaseModel):
    # required fields — must be present
    # ---------------------------------
    name: str
    steps: int
    environment: Environment = Environment()


    # optional fields — can be omitted
    # ---------------------------------
    n_robots_yellow: Optional[int] = None
    save_model: bool = True

    # reject any field not defined above
    model_config = ConfigDict(extra="forbid")

class Config(BaseModel):
    # required fields — must be present
    # ---------------------------------
    stages: list[Stage]

    # optional fields — can be omitted
    # ---------------------------------
    defaults: dict = Field(default_factory=dict)

    # reject any field not defined above
    model_config = ConfigDict(extra="forbid")


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)

# test reading the config
if __name__ == "__main__":
    config = load_config("example_config.yml")

    for stage in config.stages:
        print(stage.steps)


