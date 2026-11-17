import os
import io
import numpy as np
import pandas as pd
import lightkurve as lk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

PER_CLASS = 250          # Total 500 samples (250 planets + 250 false positives)
MAX_QUARTERS = 2         # Quarters limit for faster and stable downloading
EPOCHS = 30
BINS = 201
CACHE = "dataset_cache.npz"
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ───────── Shared preprocessing ─────────
def prep_flux(flux):
    flux = np.asarray(flux, dtype=np.float64)
    med = np.nanmedian(flux) if np.isfinite(flux).any() else 1.0
    flux = np.nan_to_num(flux, nan=med)
    flux = (flux - med) / 1e-3             # keep real transit depth (1e-3 = 0.1%)
    return np.clip(flux, -20, 20).astype(np.float32)


def prep_meta(period, snr):
    return np.array([min(period / 100.0, 1.0), min(snr / 50.0, 1.0)], dtype=np.float32)


def fixed_length(binned, period):
    """Force exactly BINS points, whatever lightkurve returns."""
    ph = np.asarray(binned.phase.value, dtype=np.float64)
    fl = np.asarray(binned.flux.value, dtype=np.float64)
    ok = np.isfinite(ph) & np.isfinite(fl)
    if ok.sum() < 20:
        return None
    grid = np.linspace(-period / 2, period / 2, BINS)
    return np.interp(grid, ph[ok], fl[ok])


# ───────── 1. Data (Using Official NASA Astroquery) ─────────
def build_dataset():
    print("📥 Querying NASA Exoplanet Archive via official API...")
    
    df = NasaExoplanetArchive.query_criteria(
        table="cumulative", 
        select="kepid,kepoi_name,koi_period,koi_time0bk,koi_model_snr,koi_disposition"
    ).to_pandas()

    df = df.rename(columns={"koi_model_snr": "koi_snr"})
    df = df.dropna(subset=["kepid", "koi_period", "koi_time0bk", "koi_disposition"])
    df["koi_snr"] = df["koi_snr"].fillna(10.0)
    
    df["koi_disposition"] = df["koi_disposition"].str.strip().str.upper()
    df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])]

    df = (df.groupby("koi_disposition", group_keys=False)
            .apply(lambda g: g.sample(n=min(PER_CLASS * 2, len(g)), random_state=SEED))
            .sample(frac=1, random_state=SEED))

    X, M, Y, IDS = [], [], [], []
    n_pos = n_neg = 0
    for _, row in df.iterrows():
        label = 1.0 if row["koi_disposition"] == "CONFIRMED" else 0.0
        if (label == 1 and n_pos >= PER_CLASS) or (label == 0 and n_neg >= PER_CLASS):
            continue
        kepid, period, t0, snr = int(row["kepid"]), float(row["koi_period"]), float(row["koi_time0bk"]), float(row["koi_snr"])
        try:
            sr = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler", author="Kepler", cadence="long")
            if len(sr) == 0:
                continue
            lc = sr[:MAX_QUARTERS].download_all(quality_bitmask="default").stitch().remove_nans()
            folded = lc.flatten(window_length=401).fold(period=period, epoch_time=t0)
            flux = fixed_length(folded.bin(bins=BINS), period)
            if flux is None:
                continue
            X.append(prep_flux(flux))
            M.append(prep_meta(period, snr))
            Y.append(label)
            IDS.append(kepid)
            n_pos += label == 1
            n_neg += label == 0
            if len(Y) % 10 == 0:
                print(f"  {len(Y)} samples processed (planets {n_pos}, false positives {n_neg})")
        except Exception:
            continue
        if n_pos >= PER_CLASS and n_neg >= PER_CLASS:
            break

    np.savez(CACHE, X=np.array(X), M=np.array(M), Y=np.array(Y), IDS=np.array(IDS))
    print(f"💾 Cached dataset to {CACHE}")


if not os.path.exists(CACHE):
    build_dataset()

d = np.load(CACHE)
X, M, Y, IDS = d["X"], d["M"], d["Y"], d["IDS"]
if len(Y) < 10:
    raise SystemExit("❌ Too few samples collected. Check internet / settings.")
print(f"✨ Loaded {len(Y)} samples | planets: {int(Y.sum())} | false positives: {int(len(Y)-Y.sum())}")

# ───────── 2. Star-wise train / validation split ─────────
stars = np.unique(IDS)
np.random.shuffle(stars)
val_stars = set(stars[: max(1, int(0.2 * len(stars)))])
is_val = np.array([k in val_stars for k in IDS])


def to_ds(mask):
    return TensorDataset(
        torch.tensor(X[mask]).unsqueeze(1).float(), 
        torch.tensor(M[mask]).float(),
        torch.tensor(Y[mask]).unsqueeze(1).float()
    )


train_dl = DataLoader(to_ds(~is_val), batch_size=16, shuffle=True)
val_dl = DataLoader(to_ds(is_val), batch_size=64)
print(f"Train samples: {(~is_val).sum()} | Validation samples: {is_val.sum()}")


# ───────── 3. Model ─────────
class AdvancedExoplanetResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.res_conv1 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
        self.res_conv2 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 50 + 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_flux, x_meta):
        x = self.pool(self.relu(self.conv1(x_flux)))
        residual = x
        out = self.relu(self.res_conv1(x))
        out = self.res_conv2(out)
        x = self.relu(out + residual)
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.cat((x, x_meta), dim=1)
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))


def auc_score(y, p):
    y, p = np.asarray(y).ravel(), np.asarray(p).ravel()
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


model = AdvancedExoplanetResNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

# ───────── 4. Training ─────────
best = float("inf")
for epoch in range(EPOCHS):
    model.train()
    tot = 0.0
    for fl, me, lb in train_dl:
        if np.random.rand() < 0.5:            # mirror augmentation for symmetry
            fl = torch.flip(fl, dims=[-1])
        fl = fl + 0.05 * torch.randn_like(fl)
        optimizer.zero_grad()
        loss = criterion(model(fl, me), lb)
        loss.backward()
        optimizer.step()
        tot += loss.item() * fl.size(0)
    tr_loss = tot / len(train_dl.dataset)

    model.eval()
    vl, preds, ys = 0.0, [], []
    with torch.no_grad():
        for fl, me, lb in val_dl:
            p = model(fl, me)
            vl += criterion(p, lb).item() * fl.size(0)
            preds.append(p.numpy())
            ys.append(lb.numpy())
    vl /= len(val_dl.dataset)
    preds, ys = np.concatenate(preds), np.concatenate(ys)
    acc = float(((preds > 0.5) == (ys > 0.5)).mean())
    scheduler.step(vl)
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | train {tr_loss:.4f} | val {vl:.4f} | acc {acc:.3f} | AUC {auc_score(ys, preds):.3f}")

    if vl < best:
        best = vl
        torch.save(model.state_dict(), "exoplanet_model.pth")

print(f"\n🎉 Done! Best validation loss {best:.4f} saved to exoplanet_model.pth")