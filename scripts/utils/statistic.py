from __future__ import annotations

import argparse
import code
import os
import re
import warnings
from glob import glob
from itertools import product

import pandas as pd
from scipy import stats

from dhgl.script_utils.configs.evalulator import (
    HGBLinkPredEvaluator,
    HGBNodeClassificationEvaluatorConfig,
    OGBEvaluator,
)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
PRECISION = 4
pd.set_option('display.float_format', lambda c: f'{c:.{PRECISION}f}')

OUTPUT_FILE = lambda root: f'summary_{os.path.basename(root)}.csv'
OUTPUT_FILE_PVALUE = lambda root: f'summary_{os.path.basename(root)}_pvalue.csv'
OUT_DIR = 'csvs'
# MACRO_F1 = 'macro_f1/test'
# MICRO_F1 = 'micro_f1/test'
# MACRO_F1_VAL = 'macro_f1/val'
# MICRO_F1_VAL = 'micro_f1/val'
LOSS_VAL = 'loss/val'

EVALUATORS = {
    'f1s': HGBNodeClassificationEvaluatorConfig.metrics,
    'acc': OGBEvaluator.metrics,
    'hgb-lp': ['roc_auc', 'mrr', 'ndcg'],
}

METRICS = None


def set_metrics(ref_df: pd.DataFrame):
    global METRICS

    def get_metrics(evaluator: str):
        return [
            *[f'{met}/test' for met in EVALUATORS[evaluator]],
            'epoch',
            LOSS_VAL,
            *[f'{met}/val' for met in EVALUATORS[evaluator]],
        ]

    for k, mets in EVALUATORS.items():
        for met in mets:
            if f'{met}/test' in ref_df.columns:
                METRICS = get_metrics(k)
                break
    if METRICS is None:
        raise NotImplementedError
    return


def aggregate(df: pd.DataFrame):
    return df.aggregate(
        {
            **{
                met: ['mean', 'std']
                for met in METRICS
            },
            'profile': 'count',
        }
    )


def reduce_csvs(root: str, filter_regex: str = None):

    def read_df(file):
        path = os.path.join(root, file)
        df = pd.read_csv(path)
        profile = '.'.join(os.path.split(file))[:-4]
        if profile.startswith('.'):
            profile = profile[1:]
        df.loc[:, 'profile'] = profile
        return df

    def find_csvs():
        for file in glob('**/*.csv', root_dir=root, recursive=True):
            path = os.path.join(root, file)
            if os.path.basename(path).startswith('_'):
                continue
            if filter_regex is not None:
                if re.search(filter_regex, path):
                    yield read_df(file)
            else:
                yield read_df(file)

    dfs = list(find_csvs())
    assert dfs, f'no csv file in dir {root} with filiter={filter_regex}'
    df = pd.concat(dfs).reset_index(drop=True)
    return df


def display_agged_df(agged_df: pd.DataFrame):

    def wrap(metric_name):
        if 'epoch' in metric_name:

            def epoch_fn(row):
                return f'{row["mean"]: >{PRECISION+1}.{PRECISION//2}f}±{row["std"]: >{PRECISION+1}.{PRECISION//2}f}'

            return epoch_fn

        def fn(row):
            return f'{row["mean"]:.{PRECISION}f}±{row["std"]:.{PRECISION}f}'

        return fn

    # COLUMNS = [
    #     MACRO_F1, MICRO_F1, 'epoch', LOSS_VAL, MACRO_F1_VAL, MICRO_F1_VAL
    # ]
    rest_df = agged_df[[
        col for col in agged_df.columns if col[0] not in METRICS
    ]]
    display_df = pd.concat(
        [agged_df[metric].apply(wrap(metric), axis=1) for metric in METRICS],
        axis=1
    )
    display_df.columns = METRICS
    display_df = pd.concat([display_df, rest_df], axis=1)
    return display_df


def main():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('env_dir', type=str)
    arg_parser.add_argument(
        '-f',
        '--filter',
        type=str,
        required=False,
    )
    arg_parser.add_argument(
        '-r',
        '--rounds',
        type=int,
        default=5,
    )
    arg_parser.add_argument(
        '-i',
        '--interactive',
        action='store_true',
    )
    arg_parser.add_argument(
        '--force',
        action='store_true',
    )
    pd.set_option('display.max_colwidth', 10000)
    args = arg_parser.parse_args()

    root_dir = os.path.abspath(args.env_dir)
    df = reduce_csvs(root_dir, args.filter)
    # df = pd.read_csv('runs_acm/lrwd_grid.csv')

    set_metrics(df)
    df = df[['profile', *METRICS]]

    agged_df = aggregate(df.groupby(['profile']))

    if not args.force:
        assert (agged_df[('profile', 'count')]
                != args.rounds).sum() == 0, display_agged_df(agged_df)
        out_df = agged_df.drop([('profile', 'count')], axis=1)
    else:
        df = pd.concat(
            [
                df[df['profile'] == profile][:args.rounds]
                for profile in agged_df.index
            ]
        )

        agged_df = aggregate(df.groupby(['profile']))
        out_df = agged_df
    # print(out_df)
    print(display_agged_df(out_df))

    if args.interactive:
        code.interact(local=dict(globals(), **locals()))
    else:
        if os.path.exists(OUT_DIR):
            out_df.to_csv(os.path.join(OUT_DIR, OUTPUT_FILE(root_dir)))

    # out_df = pvalue_macro_f1_val(
    #     agged_df,
    #     os.path.basename(root_dir),
    #     datasets=['acm', 'dblp', 'imdb'],
    #     models=['HGT', 'Simple'],
    # )
    # out_df.to_csv(os.path.join(OUT_DIR, OUTPUT_FILE_PVALUE(root_dir)))
    return


def pvalue_macro_f1_val(
    agged_df: pd.DataFrame, out_name: str, datasets: list, models: list
):

    def get_pvalue(sub_df: pd.DataFrame):
        # sub_df = sub_df[~sub_df.index.str.contains('baseline')]
        sub_df = sub_df.sort_values((MACRO_F1_VAL, 'mean'), ascending=False)
        val_f1 = sub_df[[
            (MACRO_F1_VAL, 'mean'),
            (MACRO_F1_VAL, 'std'),
            ('profile', 'count'),
        ]]
        max_row = val_f1.iloc[0].to_list()
        max_row[-1] = int(max_row[-1])

        def pvalue(row: pd.Series):
            m, s, c = row
            return stats.ttest_ind_from_stats(
                *max_row, m, s, int(c), alternative='greater'
            ).pvalue

        pvalue_df = val_f1.apply(pvalue, axis=1).to_frame('pvalue')
        sub_df = pd.concat([sub_df, pvalue_df], axis=1)
        return sub_df.drop([('profile', 'count')], axis=1)

    sub_dfs = []
    for dataset, model in product(datasets, models):
        sub_df = agged_df[agged_df.index.str.contains(model)]
        sub_df = sub_df[sub_df.index.str.contains(dataset)]
        if sub_df.empty:
            warnings.warn(f'Cannot find any row for "{model}" on "{dataset}"')
            continue
        sub_dfs.append(get_pvalue(sub_df))

    assert sub_dfs
    return pd.concat(sub_dfs, axis=0)


if __name__ == '__main__':
    main()
