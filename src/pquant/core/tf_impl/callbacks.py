from pathlib import Path
import os, csv, numpy as np
import keras

def _to_float(x):
    if x is None:            return None
    if hasattr(x, "numpy"):  x = x.numpy()
    return float(x)

def _fmt(x):                 
    return "" if x is None else f"{_to_float(x):.6f}"

class Callback:
    def on_epoch_end(self, *_, **__):     # noqa: D401
        return False                      

class CSVLogger(Callback):
    def __init__(self, path="train_history.csv"):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(
                    ["epoch", "train_loss", "val_loss", "keep_ratio"]
                )

    def on_epoch_end(self, epoch, train_loss, val_loss, model):
        from pquant import get_layer_keep_ratio
        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, _fmt(train_loss), _fmt(val_loss),
                 f"{get_layer_keep_ratio(model):.6f}"]
            )
        return False                      

class ModelCheckpoint(Callback):
    """Snapshot best validation loss only."""
    def __init__(self, path="best_weights.h5"):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.best = np.inf

    def on_epoch_end(self, _epoch, _tr, val_loss, model):
        val = _to_float(val_loss)
        if val is not None and val < self.best:
            self.best = val
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            model.save_weights(tmp)
            os.replace(tmp, self.path)
        return False

class EpochCheckpoint(Callback):
    """Save weights *every* epoch as <dir>/<prefix>_epochXXXXXX.h5."""
    def __init__(self, directory, prefix="weights"):
        self.dir = Path(directory).expanduser().resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix

    def on_epoch_end(self, epoch, *_args):
        target = self.dir / f"{self.prefix}_epoch{epoch:06d}.h5"
        tmp    = target.with_suffix(".tmp")
        _args[-1].save_weights(tmp)             
        os.replace(tmp, target)
        return False

class EarlyStopping(Callback):
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.bad = np.inf, 0

    def on_epoch_end(self, _epoch, _tr, val_loss, _model):
        val = _to_float(val_loss)
        if val is None:
            return False                       
        if val + self.min_delta < self.best:
            self.best, self.bad = val, 0
        else:
            self.bad += 1
        return self.bad >= self.patience