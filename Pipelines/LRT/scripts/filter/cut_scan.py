#!/usr/bin/env python3

from pathlib import Path
import itertools
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch


def load_model(ckpt, device=None):
    from Filter.Models.vanilla_filter import VanillaFilter
    
    device = device or 'cuda'
    
    e_ckpt = torch.load(ckpt, map_location=device)
    e_config = e_ckpt['hyper_parameters']
    e_model = VanillaFilter(e_config)
    e_model.load_state_dict(e_ckpt["state_dict"])
    e_model.to(device)
    e_model.eval()
    e_model.setup('test')

    return e_model


def test_model(model, edge_cuts, device=None):
    """
    Test model.

    @return pur, eff
    """
    device = device or 'cuda'
    
    # Initialize result statistics.
    result = {}
    for cut in edge_cuts:
        result[cut] = {
            'true': 0,
            'true_positive': 0,
            'positive': 0
        }
    
    for batch_idx, batch in enumerate(tqdm(model.test_dataloader())):
        with torch.no_grad():
            batch = batch.to(device)
            eval_result = model.shared_evaluation(
                batch, batch_idx, edge_cut=0.5
            )

            for cut in edge_cuts:
                cut_list = (eval_result['preds'] > cut)
                result[cut]['true'] += eval_result['truth'].sum().detach().cpu().item()
                result[cut]['true_positive'] += (
                    eval_result['truth'].bool() & cut_list
                ).sum().detach().cpu().float()
                result[cut]['positive'] += cut_list.sum().detach().cpu().float()
            
            # Ensure gpu resources are released.
            del batch
            del eval_result
            
            if device == 'cuda' or device == 'gpu':
                torch.cuda.empty_cache()
    
    return {cut: {
        'eff': result[cut]['true_positive'] / result[cut]['true'],
        'pur': result[cut]['true_positive'] / result[cut]['positive'],
        'n_edges': result[cut]['positive']
    } for cut in edge_cuts}


def cut_scan(model, edge_cuts, save, device=None):
    df = pd.DataFrame(columns=['edge_cut', 'pur', 'eff', 'n_edges'])

    for cut, results in test_model(model, edge_cuts, device).items():
        df = df.append({
            'edge_cut': cut, 
            'pur': float(results['pur']), 
            'eff': float(results['eff']),
            'n_edges': int(results['n_edges'])
        }, ignore_index=True)

    # Log current result.
    df.to_csv(save)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', type=Path)
    parser.add_argument('save', type=Path)
    args = parser.parse_args()

    model = load_model(args.ckpt)
    
    edge_cuts = np.arange(0.05, 1.01, 0.05)

    cut_scan(model, edge_cuts, args.save)

