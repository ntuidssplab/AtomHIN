from importlib import metadata

from pydantic import BaseModel


class SaveReason(BaseModel):

    early_stopping: bool = False
    end: bool = False
    regular: int | None = None  # epochs_per_checkpoint
    best: bool = False


_LogHistory = dict[str, float]


class CheckpointState(BaseModel):

    best_metric_criterion: str | None
    best_metrics: _LogHistory | None
    save_reason: SaveReason | None

    epoch: int

    log_history: list[_LogHistory]

    nf_version: str = 'dhgl.naive-flow:' + metadata.metadata("dhgl")["Version"]
