# Copied from HF Transformers library at https://github.com/huggingface/transformers/blob/8698b5a52598d13a4f9e7fe46457526dae967a79/src/transformers/optimization.py#L134
# With added LR floor, after cleanrl ppo_trxl.py (final_lr): annealing all the way to 0 makes
# the tail of training do essentially nothing — decay onto a floor instead.
# Restart cycles with a decaying peak LR follow the design of katsura-jp's
# CosineAnnealingWarmupRestarts (https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup),
# but implemented as a pure function of the step so it stays a LambdaLR.

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
import math


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int,
    cycle_decay: float,
    min_lr_ratio: float,
):
    cycle_length = max(1, num_training_steps // num_cycles)
    cycle = min(
        current_step // cycle_length, num_cycles - 1
    )  # leftover steps from // stay in the last cycle
    step_in_cycle = current_step - cycle * cycle_length
    peak_ratio = cycle_decay**cycle
    floor_ratio = min(
        min_lr_ratio, peak_ratio
    )  # a decayed peak below the floor would invert the decay

    if step_in_cycle < num_warmup_steps:
        warmup_start = (
            floor_ratio if cycle > 0 else 0.0
        )  # only the very first warmup ramps from 0
        return warmup_start + (peak_ratio - warmup_start) * float(
            step_in_cycle
        ) / float(max(1, num_warmup_steps))

    progress = min(
        1.0,
        float(step_in_cycle - num_warmup_steps)
        / float(max(1, cycle_length - num_warmup_steps)),
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    return floor_ratio + (peak_ratio - floor_ratio) * cosine


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    cycle_decay: float = 1,
    last_epoch: int = -1,
    min_lr_ratio: float = 0.0,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to `min_lr_ratio` times that lr, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer. With `num_cycles > 1` this warmup+decay pattern
    restarts (SGDR-style): each restart warms up from the LR floor to its peak, and each restart's peak lr is
    multiplied by `cycle_decay`.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the linear warmup phase at the start of each cycle. Must be smaller than a
            cycle's length (`num_training_steps // num_cycles`).
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of warmup+cosine-decay cycles (restarts). 1 is the plain single-cycle schedule.
        cycle_decay (`float`, *optional*, defaults to 1.0):
            The peak lr of each successive cycle is multiplied by this factor (e.g. 0.5 halves the max lr at every
            restart); expected in (0, 1]. 1.0 disables the decay.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        min_lr_ratio (`float`, *optional*, defaults to 0.0):
            The LR floor as a fraction of the initial lr; the cosine decay lands on this floor instead of 0.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    if num_cycles < 1:
        raise ValueError(f"num_cycles must be >= 1, got {num_cycles}")
    if num_warmup_steps >= max(1, num_training_steps // num_cycles):
        raise ValueError(
            f"num_warmup_steps ({num_warmup_steps}) must be smaller than each cycle's length"
            + f" ({num_training_steps} // {num_cycles} = {num_training_steps // num_cycles})"
        )

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        cycle_decay=cycle_decay,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
