import os
import typing
import dgl
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import f1_score
from xgboost import XGBClassifier

import naive_flow as nf

from dhgl import transforms
from dhgl import hgget as H
from dhgl import evaluation
from dhgl.models.TreeGNN import TreeGNN

from .legacy_config import TreeConfig, XGBConfig

Split = typing.Literal['train', 'val', 'test']


def _train_xgb(
    config: XGBConfig,
    features: np.ndarray,
    labels: np.ndarray,
    masks: dict[Split, np.ndarray],
):

    xgb_clf = XGBClassifier(
        n_estimators=config.n_estimators,
        gamma=config.gamma,
        min_child_weight=config.min_child_weight,
        colsample_bytree=config.colsample_bytree,
        max_depth=config.max_depth,
        alpha=config.alpha,
        n_jobs=-1,
        verbosity=0,
    )
    # Train & Fit the model
    xgb_clf.fit(
        features[masks['train']], labels[masks['train']], verbose=False
    )

    # tree_pred = xgb_clf.predict_proba(feats_list_np[0])
    # tree_prob = mat2tensor(device, tree_pred)
    # X_leaves = xgb_clf.apply(feats_list_np[0])

    writer = SummaryWriter(comment=config.comment)
    writer.add_text('config', nf.strfconfig(config, strformat='markdown'))
    """XGB classifer evaluation"""

    def evaluate(split: Split):
        mask = masks[split]
        feature = features[mask]
        label = labels[mask]
        pred = xgb_clf.predict(feature)
        micro_f1 = f1_score(label, pred, average='micro')
        macro_f1 = f1_score(label, pred, average='macro')
        writer.add_scalar(f'micro_f1/{split}', micro_f1)
        writer.add_scalar(f'macro_f1/{split}', macro_f1)

    evaluate('train')
    evaluate('test')
    return xgb_clf


def _reindex_leaves(x_leaves):
    leaves = x_leaves.copy()
    new_leaf_index = dict()  # Dictionary to store leaf index
    total_leaves = 0
    for c in range(
        x_leaves.shape[1]
    ):  # Iterate for each column (i.e. # trees)
        column = x_leaves[:, c]
        unique_vals = list(sorted(set(column)))
        new_idx = {v: (i + total_leaves) for i, v in enumerate(unique_vals)}
        for i, v in enumerate(unique_vals):
            leaf_id = i + total_leaves
            new_leaf_index[leaf_id] = {c: v}
        leaves[:, c] = [new_idx[v] for v in column]
        total_leaves += len(unique_vals)

    assert leaves.ravel().max() == total_leaves - 1
    return leaves, total_leaves, new_leaf_index


def _load_data(config: TreeConfig):
    hg = config.dataset.load()
    """Preprocessing"""
    # TreeGNN requires target node have order 0.
    order = [
        H.tgt_ntype(hg),
        *(ntype for ntype in hg.ntypes if ntype != H.tgt_ntype(hg))
    ]
    # features = {ntype: hg.ndata['feat'][ntype] for ntype in order}
    features = [hg.ndata['feat'][ntype] for ntype in order]

    g = transforms.to_homogeneous(hg, order=order)
    new_etype_id = len(hg.etypes)
    g: dgl.DGLGraph = dgl.add_self_loop(
        g, edge_feat_names=['_TYPE'], fill_data=new_etype_id
    )
    g.ndata['target_mask'] = g.ndata['_TYPE'] == 0
    hg.ndata.pop('feat')  # avoid misuse

    return hg, g, features


@H.use_cache()
def trainer(config: TreeConfig):
    """Trainer for the MODEL"""

    hg, g, features = _load_data(config)

    def init():
        """Initialize the MODEL"""
        xgb_clf = _train_xgb(
            config.xgb_config,
            features[0],
            H.label(hg),
            {
                'train': H.mask(hg, 'train'),
                'test': H.mask(hg, 'test')
            },
        )
        x_leaves, leaf_num, _ = _reindex_leaves(xgb_clf.apply(features[0]))
        tree_prob = xgb_clf.predict_proba(features[0])
        x_leaves = torch.from_numpy(x_leaves)
        tree_prob = torch.from_numpy(tree_prob)

        feat_dims = [data.shape[-1] for data in features]

        model = TreeGNN(
            **config.gnn_config.model_dump(),
            num_etypes=len(hg.etypes) + 1,  # +1 is for self-loop
            in_dims=feat_dims,
            num_classes=H.n_classes(hg),
            num_heads=H.n_classes(hg),
            activation=F.elu,
            residual=True,
            alpha=0.05,
            max_leaf=leaf_num,
            device=config.device,
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

        def loss_fn(logits, pred_tree, labels):
            loss1 = config.loss_fn(logits, labels)
            loss2 = config.loss_fn(pred_tree, labels)
            return (1. - config.alpha) * loss1 + config.alpha * loss2

        return model, optimizer, scheduler, loss_fn, x_leaves, tree_prob

    model, optimizer, scheduler, loss_fn, x_leaves, tree_prob = init()
    """Device Control"""
    model.to(config.device)
    features = [feature.to(config.device) for feature in features]
    tree_prob = tree_prob.to(config.device)
    x_leaves = x_leaves.to(config.device).long()
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

    def forward():
        logits, pred_tree =\
            model.forward(g, x_leaves, tree_prob, features, g.edata[dgl.ETYPE])

        return logits[g.ndata['target_mask']], pred_tree

    for epoch in tracker.range(config.epochs):

        logits: dict[Split, torch.Tensor] = {}
        losses: dict[Split, torch.Tensor] = {}

        model.train()
        logits['train'], pred_tree = forward()
        logits['train'] = logits['train'][H.mask(hg, 'train')]
        losses['train'] = loss_fn(
            logits['train'],
            pred_tree[H.mask(hg, 'train')],
            H.label(hg, 'train'),
        )

        optimizer.zero_grad()
        losses['train'].backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            model.eval()
            eval_logits, pred_tree = forward()
            logits['val'] = eval_logits[H.mask(hg, 'val')]
            logits['test'] = eval_logits[H.mask(hg, 'test')]
            losses['val'] = loss_fn(
                logits['val'],
                pred_tree[H.mask(hg, 'val')],
                H.label(hg, 'val'),
            )
            losses['test'] = loss_fn(
                logits['test'],
                pred_tree[H.mask(hg, 'test')],
                H.label(hg, 'test'),
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

    config = TreeConfig(_env_file=env_path)
    print(nf.strfconfig(config))
    best_metrics = trainer(config)
    print('best:')
    print(best_metrics)


if __name__ == '__main__':
    main()
