import os
from pathlib import Path
from typing import Tuple, Union, Dict, Optional

import pandas as pd
import torch

from .model import DeepDTA
from .trainer import Trainer


def train_deepdta(
        fps: Union[str, Path, Tuple],
        out_fp: Union[str, Path],
        split_names=("train", "val", "test"),
        column_map: Optional[Dict[str, str]] = None,
        channel: int = 32,
        protein_kernel: int = 8,
        ligand_kernel: int = 8,
        num_epochs: int = 50,
        batch_size: int = 256,
        lr: float = 0.001,
):
    # this CSV file has 4 columns, protein, ligands, affinity, split.

    if isinstance(fps, Tuple):
        dfs = []
        assert len(fps) == len(split_names)
        for fp, name in zip(*(fps, split_names)):
            df = pd.read_csv(fp)
            df["split"] = name
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(fps)
    df.rename(columns=column_map, inplace=True)

    idx = []
    for name in split_names:
        idx.append(df[df["split"] == name].index.values)
    while len(idx) < 3:
        idx.append(idx[-1])

    torch.manual_seed(42)
    res_fp = Path(out_fp)
    res_fp.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(DeepDTA, channel, protein_kernel, ligand_kernel, df,
                      idx[0], idx[1], idx[2], res_fp / "training.log")
    trainer.train(num_epochs=num_epochs, batch_size=batch_size, lr=lr, save_path=res_fp / 'deepdta.pt')

    test = pd.read_csv(f"test-result-prk{protein_kernel}-ldk{ligand_kernel}.txt", header=None)
    test.columns = ["Pred", "Label"]
    test.to_csv(res_fp / "test_predictions.csv", index=False)
    os.remove(f"test-result-prk{protein_kernel}-ldk{ligand_kernel}.txt")
