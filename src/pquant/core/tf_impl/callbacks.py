# -*- coding: utf-8 -*-
# @Author: Arghya Ranjan Das
# file: src/pquant/core/tf_impl/callbacks.py
# modified by:

import keras
import os
import csv
from pathlib import Path
import numpy as np

class Callback:
    def on_epoch_end(self, epoch, train_loss, val_loss, model, **kwargs):
        return False

class CSVLogger(Callback):
    def __init__(self, path="train_history.csv"):
        self.path = path
        if not os.path.isfile(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["epoch", "train_loss", "val_loss", "keep_ratio"]
                )

    def on_epoch_end(self, epoch, train_loss, val_loss, model):
        from pquant import get_layer_keep_ratio
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                 f"{get_layer_keep_ratio(model):.6f}"]
            )
        return False

class ModelCheckpoint(Callback):
    def __init__(self, path="best_weights.h5"):
        self.path, self.best = path, np.inf

    def on_epoch_end(self, epoch, train_loss, val_loss, model):
        if val_loss < self.best:
            self.best = val_loss
            tmp = self.path + ".tmp"
            model.save_weights(tmp)
            os.replace(tmp, self.path)
        return 
    
class EpochCheckpoint(Callback):
    """
    Save weights at *every* epoch as  <dir>/<prefix>_epochXXXX.h5
    Files are written atomically; nothing is removed.
    """
    def __init__(self, directory: Path, prefix: str = "weights"):
        self.dir = Path(directory).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix

    def on_epoch_end(self, epoch, train_loss, val_loss, model):
        fname = f"{self.prefix}_epoch{epoch:06d}.h5"
        target = self.dir / fname
        tmp    = target.with_suffix(".tmp")
        model.save_weights(tmp)      # atomic: write→rename
        os.replace(tmp, target)
        return False 
    
class EarlyStopping(Callback):
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.bad = np.inf, 0

    def on_epoch_end(self, epoch, train_loss, val_loss, model):
        if val_loss + self.min_delta < self.best:
            self.best, self.bad = val_loss, 0
        else:
            self.bad += 1
        return self.bad >= self.patience