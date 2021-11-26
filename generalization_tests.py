import torch
import os
import json

from argparse import ArgumentParser
from logistics_helpers import all_logistics, all_logistics_test


def main():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--z_file_path', type=str, required=True)
    parser.add_argument('--y_file_path', type=str, required=True)
    parser.add_argument('--c_file_path', type=str, required=True)
    parser.add_argument('--y_pred_file_path', type=str, default=None)
    parser.add_argument('--include_test', action='store_true')
    parser.add_argument('--sample', type=int, default=1)
    args = parser.parse_args()

    z_splits = torch.load(os.path.join(args.root_dir, args.z_file_path))
    y_splits = torch.load(os.path.join(args.root_dir, args.y_file_path))
    c_splits = torch.load(os.path.join(args.root_dir, args.c_file_path))
    if args.y_pred_file_path is not None:
        y_pred_splits = torch.load(os.path.join(args.root_dir, args.y_pred_file_path))
    else:
        y_pred_splits = None

    if args.include_test:
        logistics = all_logistics_test(z_splits, c_splits, y_splits, sample=args.sample)
    else:
        logistics = all_logistics(z_splits, c_splits, y_splits, sample=args.sample)

    logistics['G1'] = logistics['val_on_val']
    logistics['G2'] = logistics['trainval_on_val']

    if y_pred_splits is not None:
        logistics['G0'] = (1.0 * (y_splits['id_val'].argmax(axis=-1) == y_pred_splits['id_val'].argmax(axis=-1))).mean()
        logistics['G3'] = (1.0 * (y_splits['val'].argmax(axis=-1) == y_pred_splits['val'].argmax(axis=-1))).mean()

    logistics['I0'] = logistics['c_train']
    logistics['I1'] = logistics['c_val']
    per_class = torch.tensor(list(logistics['c_perclass'].values()))
    logistics['I2'] = torch.mean(per_class).item()

    if args.include_test:
        logistics['G1_test'] = logistics['test_on_test']
        logistics['G2_test'] = logistics['traintest_on_test']

        if y_pred_splits is not None:
            logistics['G3_test'] = \
                (1.0 * (y_splits['test'].argmax(axis=-1) == y_pred_splits['test'].argmax(axis=-1))).mean()

        logistics['I1_test'] = logistics['c_test']
        per_class = torch.tensor(list(logistics['c_perclass_test'].values()))
        logistics['I2_test'] = torch.mean(per_class).item()

    with(open(os.path.join(args.out_dir, 'generalization_test_results.json'), "w")) as f:
        json.dump(logistics, f, indent=True)


if __name__ == '__main__':
    main()
