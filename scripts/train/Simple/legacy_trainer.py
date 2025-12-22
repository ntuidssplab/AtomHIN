import os
from typing import Literal
import dgl
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from naive_flow.tracker import TrackerConfig
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import naive_flow as nf

from dhgl import hgget as H
from dhgl import evaluation
from dhgl.models import SimpleHGN

from dhgl.script_utils.configs.batch import BatchConfig
from dhgl.script_utils.configs.dataset import ACMConfig, DBLPConfig, IMDBConfig
from dhgl.script_utils.configs.loss import BCEWithLogits, CrossEntropy, SoftLabelCE


class SimpleConfig(BaseSettings):

    model_config = SettingsConfigDict(
        env_nested_delimiter='__', extra='allow', frozen=True
    )

    name: Literal['Simple'] = 'Simple'

    ###################
    # DATASET CONFIGS #
    ###################

    dataset: ACMConfig | DBLPConfig | IMDBConfig = Field(discriminator='name')

    #######################
    # myGAT MODEL CONFIGS   #
    #######################

    edge_embedding_dim: int

    hidden_dim: int

    num_layers: int

    num_heads: int
    """Number of attention heads"""

    dropout: float
    negative_slope: float

    loss_fn: CrossEntropy | BCEWithLogits | SoftLabelCE = Field(
        discriminator='name'
    )

    ####################
    # TRAINING CONFIGS #
    ####################
    device: str = 'cuda:0'
    epochs: int

    lr: float
    weight_decay: float
    max_lr_scale: float
    pct_start_epoch: int

    tracker_config: TrackerConfig

    batch_config: BatchConfig | None = None


def _load_data(config: SimpleConfig):
    """Load data"""
    hg = config.dataset.load()
    """Task-specifiic transform"""
    g = dgl.to_homogeneous(hg)
    new_etype_id = len(hg.etypes)
    g: dgl.DGLGraph = dgl.add_self_loop(
        g, edge_feat_names=['_TYPE'], fill_data=new_etype_id
    )

    g.ndata['target_mask'] = g.ndata['_TYPE'] == hg.get_ntype_id(
        H.tgt_ntype(hg)
    )

    return hg, g


@H.use_cache()
def trainer(config: SimpleConfig):
    """Trainer for the MODEL"""

    hg, g = _load_data(config)
    """Initialize the MODEL"""
    model = SimpleHGN(
        edge_dim=config.edge_embedding_dim,
        num_etypes=len(hg.etypes) + 1,  # original etypes + self-loop
        in_dims=[data.shape[-1] for data in hg.ndata['feat'].values()],
        num_hidden=config.hidden_dim,
        num_classes=H.n_classes(hg),
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        activation=F.elu,
        feat_drop=config.dropout,
        attn_drop=config.dropout,
        negative_slope=config.negative_slope,
        residual=True,
        alpha=0.05,
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
    g = g.to(config.device)
    hg = hg.to(config.device)
    """Init tracker"""
    tracker = nf.tracker.SimpleTracker(
        model,
        optimizer,
        scheduler,
        **dict(config.tracker_config),
        from_checkpoint=nf.tracker.checkpoint.parse_args(),
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

    def forward():
        logits = model.forward(
            g, list(hg.ndata['feat'].values()), g.edata[dgl.ETYPE]
        )
        return logits[g.ndata['target_mask']]

    for epoch in tracker.range(config.epochs):

        logits: dict[Split, torch.Tensor] = {}
        losses: dict[Split, torch.Tensor] = {}

        model.train()
        logits['train'] = forward()[H.mask(hg, 'train')]
        losses['train'] = config.loss_fn(logits['train'], H.label(hg, 'train'))

        optimizer.zero_grad()
        losses['train'].backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            model.eval()
            eval_logits = forward()
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

    best_metrics = tracker.get_best_scalars()
    writer.add_hparams(
        {'best': config.tracker_config.comment},
        dict((f'best/{name}', v) for name, v in best_metrics.items()),
        run_name='.',
    )
    nf.dump_config(config, os.path.join(tracker.log_dir, 'config.env'))
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

    config = SimpleConfig(_env_file=env_path)
    print(nf.strfconfig(config))
    best_metrics = trainer(config)
    print('best:')
    print(best_metrics)


if __name__ == '__main__':
    main()
