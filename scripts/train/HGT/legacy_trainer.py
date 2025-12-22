import os
from typing import Literal
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from naive_flow.tracker import TrackerConfig
import naive_flow as nf

from dhgl.script_utils import filter_env_private_fields
# from dhgl.evaluation import find_best_thresholds
from dhgl import evaluation
from dhgl import hgget as H
from dhgl.models import HGT
from dhgl.script_utils.configs.dataset import ACMConfig, DBLPConfig, IMDBConfig
from dhgl.script_utils.configs.loss import BCEWithLogits, CrossEntropy, SoftLabelCE


@filter_env_private_fields
class HGTConfig(BaseSettings):

    model_config = SettingsConfigDict(
        env_nested_delimiter='__', extra='forbid', frozen=True
    )

    name: Literal['HGT'] = 'HGT'

    ###################
    # DATASET CONFIGS #
    ###################

    dataset: ACMConfig | DBLPConfig | IMDBConfig = Field(discriminator='name')

    #######################
    # MODEL CONFIGS   #
    #######################
    hidden_dim: int

    num_layers: int
    """Number of Layer"""

    num_heads: int
    """Number of attention heads"""

    ####################
    # TRAINING CONFIGS #
    ####################
    device: str = 'cuda:0'
    epochs: int

    lr: float
    weight_decay: float
    max_lr_scale: float
    pct_start_epoch: int

    dropout: float

    loss_fn: CrossEntropy | BCEWithLogits | SoftLabelCE = Field(
        discriminator='name'
    )

    tracker_config: TrackerConfig


@H.use_cache()
def trainer(config: HGTConfig):
    """Trainer for the MODEL"""
    """Initialize the selected dataset"""

    hg = config.dataset.load()
    """Initialize the MODEL"""
    model = HGT(
        n_hidden=config.hidden_dim,
        etypes=hg.etypes,
        feat_shapes={
            ntype: data.shape[-1]
            for ntype, data in hg.ndata['feat'].items()
        },
        n_out=H.n_classes(hg),
        n_layers=config.num_layers,
        n_heads=config.num_heads,
        use_norm=True,
        dropout=config.dropout,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = OneCycleLR(
        optimizer,
        total_steps=config.epochs,
        max_lr=config.lr * config.max_lr_scale,
        pct_start=config.pct_start_epoch / config.epochs,
    )
    """Device Control"""
    model.to(config.device)
    hg = hg.to(config.device)
    """Init tracker"""
    tracker = nf.tracker.SimpleTracker(
        model,
        optimizer,
        scheduler,
        **dict(config.tracker_config),
        from_checkpoint=nf.tracker.checkpoint.parse_args(),
        # ^^ use cml args to decide whether start from checkpoint
    )

    writer = SummaryWriter(
        log_dir=tracker.log_dir, purge_step=tracker.start_epoch
    )
    writer = tracker.register_summary_writer(writer)
    tracker.register_scalar('macro_f1/val', 'ratio', for_early_stopping=True)
    tracker.register_scalar('*f1*', 'ratio')
    writer.add_text(
        'config', nf.strfconfig(config, strformat='markdown'),
        tracker.start_epoch
    )
    """Start Training Loop"""

    Split = Literal['train', 'val', 'test']
    for epoch in tracker.range(config.epochs):

        logits: dict[Split, torch.Tensor] = {}
        losses: dict[Split, torch.Tensor] = {}

        model.train()
        logits['train'] = model.forward(hg, hg.ndata['feat'], H.tgt_ntype(hg))
        logits['train'] = logits['train'][H.mask(hg, 'train')]
        losses['train'] = config.loss_fn(logits['train'], H.label(hg, 'train'))
        optimizer.zero_grad()
        losses['train'].backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            model.eval()
            eval_logits = model.forward(hg, hg.ndata['feat'], H.tgt_ntype(hg))
            logits['val'] = eval_logits[H.mask(hg, 'val')]
            logits['test'] = eval_logits[H.mask(hg, 'test')]
            losses['val'] = config.loss_fn(logits['val'], H.label(hg, 'val'))
            losses['test'] = config.loss_fn(
                logits['test'], H.label(hg, 'test')
            )

        f1_scores = evaluation.node_classification_eval(
            *[
                (logits[split], H.label(hg, split))
                for split in ['train', 'val', 'test']
            ],
            multi_label_thresholds='macro_f1',
        )

        for split, result in zip(['train', 'val', 'test'], f1_scores):
            writer.add_scalar(f'loss/{split}', losses[split].item(), epoch)
            writer.add_scalar(f'macro_f1/{split}', result.macro_f1, epoch)
            writer.add_scalar(f'micro_f1/{split}', result.micro_f1, epoch)
    """Training Loop Ends, some checkpoints should have been saved"""

    nf.dump_config(config, os.path.join(tracker.log_dir, 'config.env'))
    best_metrics = tracker.get_best_scalars()
    writer.add_hparams(
        {'best': config.tracker_config.comment},
        dict((f'best/{name}', v) for name, v in best_metrics.items()),
        run_name='.',
    )
    return best_metrics


def main():
    arg_parser = nf.tracker.get_default_arg_parser()
    arg_parser.add_argument(
        '-e',
        '--env',
        type=str,
        required=True,
        help='path to the .env file to use as config',
    )
    args = arg_parser.parse_args()
    env_path = args.env
    assert os.path.isfile(env_path), 'No env file found'

    config = HGTConfig(_env_file=env_path)
    print(nf.strfconfig(config))
    best_metrics = trainer(config)
    print('best:')
    print(best_metrics)


if __name__ == '__main__':
    main()
