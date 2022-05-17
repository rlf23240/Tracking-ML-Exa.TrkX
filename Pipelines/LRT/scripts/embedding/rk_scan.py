#!/usr/bin/env python3

from pathlib import Path
import itertools
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch


def load_embedding_model(ckpt, device=None):
    from Embedding.Models.layerless_embedding import LayerlessEmbedding
    
    device = device or 'cuda'
    
    e_ckpt = torch.load(ckpt, map_location=device)
    e_config = e_ckpt['hyper_parameters']
    e_model = LayerlessEmbedding(e_config)
    e_model.load_state_dict(e_ckpt["state_dict"])
    e_model.to(device)
    e_model.eval()
    e_model.setup('test')

    return e_model


def test_embedding_model(model, r, knn, device=None):
    """
    Test model.

    @return pur, eff
    """
    device = device or 'cuda'


    cluster_true = 0
    cluster_true_positive = 0
    cluster_positive = 0

    for batch_idx, batch in enumerate(tqdm(model.test_dataloader())):
        with torch.no_grad():
            batch = batch.to(device)
            eval_result = model.shared_evaluation(
                batch, batch_idx, knn_radius=r, knn_num=knn
            )
            
            cluster_true += eval_result['truth_graph'].shape[1]
            cluster_true_positive += eval_result['truth'].sum().detach().cpu().item()
            cluster_positive += len(eval_result['preds'][0])
            
            # Ensure gpu resources are released.
            del batch
            del eval_result
            
            if device == 'cuda' or device == 'gpu':
                torch.cuda.empty_cache()
    
    eff = cluster_true_positive / cluster_true
    pur = cluster_true_positive / cluster_positive

    return pur, eff

def rk_scan(model, rs, ks, save, device=None):
    df = pd.DataFrame(columns=['r', 'knn', 'pur', 'eff'])

    for r, knn in itertools.product(rs, ks):
        print(f'r={r}, knn={knn}')
        pur, eff = test_embedding_model(model, r, int(knn), device)

        df = df.append({
            'r': r, 
            'knn': knn, 
            'pur': pur, 
            'eff': eff
        }, ignore_index=True)

        # Log current result.
        df.to_csv(save)

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', type=Path)
    parser.add_argument('save', type=Path)
    args = parser.parse_args()

    model = load_embedding_model(args.ckpt)
    
    r_test = np.arange(0.1, 1.1, 0.1)
    knn_test = np.arange(100, 501, 100)

    rk_scan(model, r_test, knn_test, args.save)

