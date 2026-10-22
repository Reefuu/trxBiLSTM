"""TRXUSDT: 25 candle 1h -> Close candle berikutnya; PyTorch + sklearn.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version as package_version
from pathlib import Path
from time import perf_counter, sleep
import json
import platform
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

FEATURES = ["Open", "High", "Low", "Close", "Volume"]
TARGET = "Close"
SYMBOL = "TRXUSDT"
INTERVAL = "1h"
WINDOW_SIZE = 25
FORECAST_HORIZON = 1
HOUR = pd.Timedelta(hours=1)
BASE_URL = "https://data-api.binance.vision"
RAW_COLUMNS = [
    "Open_time", "Open", "High", "Low", "Close", "Volume", "Close_time",
    "Quote_asset_volume", "Number_of_trades", "Taker_buy_base_volume",
    "Taker_buy_quote_volume", "Ignore",
]


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 32
    learning_rate: float = 0.001
    max_epochs: int = 100
    patience: int = 10
    gradient_clip: float = 1.0
    seed: int = 42
    window_size: int = WINDOW_SIZE
    horizon: int = FORECAST_HORIZON

    def __post_init__(self):
        if self.window_size != 25 or self.horizon != 1:
            raise ValueError("Paket ini ditetapkan untuk window 25 dan horizon 1 candle 1h.")
        if min(self.hidden_size, self.num_layers, self.batch_size, self.max_epochs, self.patience) < 1:
            raise ValueError("Ukuran model, batch, epoch, dan patience harus positif.")
        if not 0 <= self.dropout < 1 or self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("Dropout, learning rate, atau gradient clipping tidak valid.")
        if self.num_layers == 1 and self.dropout != 0:
            raise ValueError("Dropout antarlapisan LSTM hanya dipakai jika num_layers > 1.")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def file_sha256(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_times(values):
    """Membaca ISO-8601 serta epoch s/ms/us/ns; ISO tanpa zona dianggap UTC."""
    s = pd.Series(values).copy()
    numeric = pd.to_numeric(s, errors="coerce")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")
    magnitude = numeric.abs()
    for unit, mask in [
        ("s", magnitude < 1e11),
        ("ms", magnitude.between(1e11, 1e14, inclusive="left")),
        ("us", magnitude.between(1e14, 1e17, inclusive="left")),
        ("ns", magnitude >= 1e17),
    ]:
        mask = mask & numeric.notna()
        if mask.any():
            out.loc[mask] = pd.to_datetime(numeric.loc[mask], unit=unit, utc=True, errors="coerce")
    mask = numeric.isna() & s.notna()
    if mask.any():
        out.loc[mask] = pd.to_datetime(s.loc[mask], format="mixed", utc=True, errors="coerce")
    return out


def validate_frame(frame, *, strict=False, as_of=None):
    """Training: catat dan keluarkan baris invalid. Inferensi: tolak baris invalid.

    Duplikasi berkonflik selalu ditolak; data tidak diinterpolasi. Candle yang
    belum selesai selalu dikeluarkan. Celah tetap ada dan diperiksa saat window.
    """
    df = frame.copy().reset_index(drop=True)
    missing = sorted(set(["Open_time"] + FEATURES) - set(df.columns))
    if missing:
        raise ValueError("Kolom wajib hilang: " + ", ".join(missing))
    if df.empty:
        raise ValueError("Dataset kosong.")
    if df.columns.duplicated().any():
        raise ValueError("Nama kolom ganda ditemukan.")
    report = {"rows_input": len(df), "invalid_rows_removed": 0,
              "duplicates_removed": 0, "unfinished_rows_removed": 0,
              "reordered": False, "gap_count": 0}
    df["Open_time"] = parse_times(df["Open_time"])
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    if "Close_time" in df:
        df["Close_time"] = parse_times(df["Close_time"])
    prices = df[["Open", "High", "Low", "Close"]]
    invalid = (~np.isfinite(df[FEATURES]).all(axis=1) | df["Open_time"].isna()
               | (prices <= 0).any(axis=1) | (df["Volume"] < 0)
               | (df["High"] < df[["Open", "Close", "Low"]].max(axis=1))
               | (df["Low"] > df[["Open", "Close", "High"]].min(axis=1)))
    invalid |= df["Open_time"].ne(df["Open_time"].dt.floor("h"))
    if "Close_time" in df:
        # Binance menutup candle tepat sebelum batas jam berikutnya (ms/us).
        earliest = df["Open_time"] + HOUR - pd.Timedelta(milliseconds=1)
        latest = df["Open_time"] + HOUR
        invalid |= df["Close_time"].isna() | (df["Close_time"] < earliest) | (df["Close_time"] >= latest)
    bad_rows = (np.flatnonzero(invalid.to_numpy()) + 2).tolist()
    if bad_rows and strict:
        raise ValueError(f"Data waktu/OHLCV tidak valid pada baris CSV {bad_rows[:10]}.")
    report["invalid_rows_removed"] = len(bad_rows)
    report["invalid_csv_rows_first_20"] = bad_rows[:20]
    df = df.loc[~invalid].copy()
    if df.empty:
        raise ValueError("Tidak ada baris valid setelah pemeriksaan waktu dan OHLCV.")
    duplicate = df["Open_time"].duplicated(keep=False)
    if duplicate.any():
        if strict:
            raise ValueError("Open_time duplikat; selesaikan tanggal/jam ganda sebelum inferensi.")
        cols = FEATURES + (["Close_time"] if "Close_time" in df else [])
        conflicts = df.loc[duplicate].groupby("Open_time")[cols].nunique(dropna=False).gt(1).any(axis=1)
        if conflicts.any():
            raise ValueError(f"Duplikasi waktu berkonflik: {conflicts[conflicts].index[0]}.")
        report["duplicates_removed"] = int(df["Open_time"].duplicated().sum())
        df = df.drop_duplicates("Open_time", keep="first")
    if not df["Open_time"].is_monotonic_increasing:
        if strict:
            raise ValueError("Open_time tidak berurutan; urutkan CSV menaik sebelum inferensi.")
        df = df.sort_values("Open_time")
        report["reordered"] = True
    now = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    complete = df["Open_time"] + HOUR <= now
    report["unfinished_rows_removed"] = int((~complete).sum())
    df = df.loc[complete].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("Belum ada candle satu jam yang sudah selesai.")
    if "Close_time" not in df:
        df["Close_time"] = df["Open_time"] + HOUR - pd.Timedelta(milliseconds=1)
    report["gap_count"] = int(df["Open_time"].diff().dropna().ne(HOUR).sum())
    report["rows_clean"] = len(df)
    report["first_open_utc"] = df["Open_time"].iloc[0].isoformat()
    report["last_open_utc"] = df["Open_time"].iloc[-1].isoformat()
    report["as_of_utc"] = now.isoformat()
    return df[["Open_time"] + FEATURES + ["Close_time"]], report


def request_json(session, endpoint, params=None, max_retries=5):
    import requests
    last_error = None
    for attempt in range(max_retries):
        try:
            response = session.get(BASE_URL + endpoint, params=params, timeout=30)
            if response.status_code in (418, 429):
                retry_after = response.headers.get("Retry-After", "")
                wait = float(retry_after) if retry_after.isdigit() else min(2 ** attempt, 30)
                if attempt < max_retries - 1:
                    sleep(wait)
                    continue
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "code" in data:
                raise ValueError(f"Binance: {data.get('msg', data['code'])}")
            return data
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < max_retries - 1:
                sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Tidak berhasil mengambil {endpoint}; periksa koneksi atau gunakan CSV.") from last_error


def fetch_hourly(data_dir="data", *, end_time_ms=None):
    """Unduh seluruh histori 1h sampai cutoff tetap; simpan RAW dan metadata."""
    import requests
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        session.headers.update({"User-Agent": "TRXUSDT-Academic-Research/1.0"})
        server_ms = int(request_json(session, "/api/v3/time")["serverTime"])
        cutoff = min(server_ms, int(end_time_ms)) if end_time_ms is not None else server_ms
        rows, start, page = [], 0, 1
        while start <= cutoff:
            batch = request_json(session, "/api/v3/klines", {
                "symbol": SYMBOL, "interval": INTERVAL, "startTime": start,
                "endTime": cutoff, "timeZone": "0", "limit": 1000,
            })
            if not isinstance(batch, list):
                raise ValueError("Format respons klines tidak sesuai.")
            if not batch:
                break
            rows.extend(batch)
            print(f"Halaman {page}: {len(batch):,} baris; total {len(rows):,}")
            next_start = int(batch[-1][0]) + 1
            if next_start <= start:
                raise RuntimeError("Pagination tidak bergerak maju.")
            start, page = next_start, page + 1
            if len(batch) < 1000:
                break
            sleep(0.1)
    if not rows:
        raise ValueError("Data klines kosong.")
    raw = pd.DataFrame(rows, columns=RAW_COLUMNS)
    raw_path = data_dir / "TRXUSDT_1h_raw.csv"
    raw.to_csv(raw_path, index=False)
    metadata = {"source": BASE_URL + "/api/v3/klines", "method": "Binance Spot REST API",
                "symbol": SYMBOL, "interval": INTERVAL, "timezone": "UTC",
                "collection_server_time_utc": pd.to_datetime(server_ms, unit="ms", utc=True).isoformat(),
                "cutoff_utc": pd.to_datetime(cutoff, unit="ms", utc=True).isoformat(),
                "raw_rows": len(raw), "raw_sha256": file_sha256(raw_path)}
    write_json(data_dir / "collection_metadata.json", metadata)
    return raw, metadata


def eligible_targets(df, window=WINDOW_SIZE):
    """Syarat window: 25 input DAN 1 target benar-benar berjarak 1h."""
    if len(df) <= window:
        raise ValueError(f"Training memerlukan lebih dari {window} candle lengkap.")
    # Tidak membentuk fitur/target lebih dulu: hanya periksa kontinuitas waktu.
    breaks = df["Open_time"].diff().ne(HOUR).to_numpy(dtype=np.int64)
    breaks[0] = 0
    cumulative = np.cumsum(breaks)
    candidates = np.arange(window, len(df))
    return candidates[(cumulative[candidates] - cumulative[candidates - window]) == 0]


class WindowDataset(Dataset):
    """Window dibuat saat dibaca; seluruh N x 25 x 5 tidak disalin ke RAM."""
    def __init__(self, x_scaled, y_scaled, target_indices, window=WINDOW_SIZE):
        self.x = np.ascontiguousarray(x_scaled, dtype=np.float32)
        self.y = np.ascontiguousarray(y_scaled, dtype=np.float32)
        self.targets = np.asarray(target_indices, dtype=np.int64)
        self.window = window

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        t = self.targets[index]
        return torch.from_numpy(self.x[t - self.window:t]), torch.from_numpy(self.y[t])


@dataclass
class PreparedData:
    frame: pd.DataFrame
    x_scaler: MinMaxScaler
    y_scaler: MinMaxScaler
    datasets: dict
    target_indices: dict
    split_table: pd.DataFrame
    scaler_fit_last_index: int


def prepare_data(df):
    """70/15/15 berdasarkan urutan timestamp TARGET sampel yang layak.

    Batas sampel ditentukan sebelum fit scaler dan pembentukan tensor.
    Konteks validasi/uji boleh berasal dari periode sebelumnya (rolling 1-step).
    """
    targets = eligible_targets(df)
    n = len(targets)
    n_train, n_val = int(n * 0.70), int(n * 0.15)
    if min(n_train, n_val, n - n_train - n_val) < 1:
        raise ValueError("Sampel kontinu terlalu sedikit untuk pembagian 70/15/15.")
    groups = {"train": targets[:n_train], "validation": targets[n_train:n_train + n_val],
              "test": targets[n_train + n_val:]}
    # Union baris yang benar-benar muncul sebagai fitur/target training.
    used = np.zeros(len(df), dtype=bool)
    for t in groups["train"]:
        used[t - WINDOW_SIZE:t + 1] = True
    train_rows = df.loc[used]
    x_scaler = MinMaxScaler(feature_range=(0, 1), clip=False).fit(train_rows[FEATURES].to_numpy())
    y_scaler = MinMaxScaler(feature_range=(0, 1), clip=False).fit(train_rows[[TARGET]].to_numpy())
    x = x_scaler.transform(df[FEATURES].to_numpy()).astype(np.float32)
    y = y_scaler.transform(df[[TARGET]].to_numpy()).astype(np.float32)
    datasets = {name: WindowDataset(x, y, indices) for name, indices in groups.items()}
    rows = []
    for name, indices in groups.items():
        rows.append({"split": name, "samples": len(indices), "percentage": 100 * len(indices) / n,
                     "first_target_utc": df["Open_time"].iloc[indices[0]].isoformat(),
                     "last_target_utc": df["Open_time"].iloc[indices[-1]].isoformat()})
    return PreparedData(df.copy(), x_scaler, y_scaler, datasets, groups,
                        pd.DataFrame(rows), int(groups["train"][-1]))


class BiLSTMRegressor(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.lstm = nn.LSTM(input_size=len(FEATURES), hidden_size=config.hidden_size,
                            num_layers=config.num_layers, batch_first=True,
                            bidirectional=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, 1)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (WINDOW_SIZE, len(FEATURES)):
            raise ValueError("Input model harus berbentuk (batch, 25, 5).")
        _, (h_n, _) = self.lstm(x)
        # Final hidden state kedua arah pada LAPISAN TERAKHIR.
        # output[:, -1, :] tidak berisi final backward state yang sama.
        context = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.fc(context)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


@dataclass
class FitResult:
    config: ModelConfig
    state_dict: dict
    history: pd.DataFrame
    best_epoch: int
    validation_rmse_usdt: float
    training_device: str
    training_seconds: float


def fit_model(data: PreparedData, config: ModelConfig, *, device=None, verbose=True):
    """Melihat training dan validation saja. Checkpoint terbaik dipulihkan."""
    started = perf_counter()
    set_seed(config.seed)
    device = choose_device() if device is None else torch.device(device)
    model = BiLSTMRegressor(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    train_loader = loader(data.datasets["train"], config.batch_size)
    val_loader = loader(data.datasets["validation"], config.batch_size)
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                raise ValueError("Loss training tidak finite; periksa data/learning rate.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip)
            optimizer.step()
            train_sum += loss.item() * len(xb)
        model.eval()
        val_sum = 0.0
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_sum += criterion(model(xb), yb).item() * len(xb)
        train_mse = train_sum / len(data.datasets["train"])
        val_mse = val_sum / len(data.datasets["validation"])
        if not np.isfinite(val_mse):
            raise ValueError("Loss validation tidak finite.")
        history.append({"epoch": epoch, "train_mse_scaled": train_mse, "val_mse_scaled": val_mse})
        if verbose:
            print(f"Epoch {epoch:03d}: train MSE={train_mse:.8f}; val MSE={val_mse:.8f}")
        if val_mse < best_loss:
            best_loss, best_epoch, stale = val_mse, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config.patience:
            if verbose:
                print(f"Early stopping; checkpoint epoch {best_epoch} dipilih.")
            break
    rmse_usdt = float(np.sqrt(best_loss) / data.y_scaler.scale_[0])
    return FitResult(config, best_state, pd.DataFrame(history), best_epoch, rmse_usdt,
                     str(device), perf_counter() - started)


def tune_model(data, base_config, *, tune_dropout=True, log_dir=None, verbose=True):
    """Penalaan bertahap sesuai kandidat proposal; data uji tidak diakses.

    1) hidden/layers; 2) batch/lr; 3) dropout jika arsitektur terbaik >1 lapisan.
    Jika tidak menala, gunakan fit_model langsung.
    """
    cache, records = {}, []
    best = None
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    def run(config, stage):
        nonlocal best
        key = json.dumps(asdict(config), sort_keys=True)
        if key in cache:
            return
        print(f"\nTahap {stage}, kandidat {len(cache) + 1}: {asdict(config)}")
        result = fit_model(data, config, verbose=verbose)
        cache[key] = True
        record = {"trial": len(cache), "stage": stage, **asdict(config),
                  "best_epoch": result.best_epoch, "val_rmse_usdt": result.validation_rmse_usdt}
        records.append(record)
        if log_dir is not None:
            pd.DataFrame(records).to_csv(log_dir / "experiments.csv", index=False)
            result.history.to_csv(log_dir / f"trial_{len(cache):02d}_history.csv", index=False)
        score = (result.validation_rmse_usdt, result.config.hidden_size,
                 result.config.num_layers, result.config.batch_size)
        old_score = None if best is None else (best.validation_rmse_usdt, best.config.hidden_size,
                                               best.config.num_layers, best.config.batch_size)
        if best is None or score < old_score:
            best = result
            if log_dir is not None:
                torch.save({"state_dict": best.state_dict, "config": asdict(best.config)},
                           log_dir / "best_validation.pt")

    for hidden in (32, 64, 128):
        for layers in (1, 2):
            run(replace(base_config, hidden_size=hidden, num_layers=layers, dropout=0.0,
                        batch_size=32, learning_rate=0.001), "arsitektur")
    architecture = best.config
    for batch in (16, 32, 64):
        for lr in (0.0001, 0.0005, 0.001):
            run(replace(architecture, batch_size=batch, learning_rate=lr), "batch_lr")
    selected = best.config
    if tune_dropout and selected.num_layers > 1:
        for dropout in (0.0, 0.2, 0.3):
            run(replace(selected, dropout=dropout), "dropout")
    return best, pd.DataFrame(records)


def regression_metrics(actual, predicted):
    a, p = np.asarray(actual).reshape(-1), np.asarray(predicted).reshape(-1)
    if len(a) == 0 or len(a) != len(p) or (a <= 0).any() or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("Target/prediksi untuk metrik tidak valid.")
    return {"mae_usdt": float(mean_absolute_error(a, p)),
            "rmse_usdt": float(np.sqrt(mean_squared_error(a, p))),
            "mape_percent": float(mean_absolute_percentage_error(a, p) * 100)}


def evaluate_test(data, fit):
    """Panggil sesudah konfigurasi dikunci; evaluasi rolling satu langkah."""
    model = BiLSTMRegressor(fit.config)
    model.load_state_dict(fit.state_dict)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for xb, _ in loader(data.datasets["test"], fit.config.batch_size):
            predictions.append(model(xb).numpy())
    predicted = data.y_scaler.inverse_transform(np.concatenate(predictions)).ravel()
    indices = data.target_indices["test"]
    frame = data.frame.iloc[indices]
    actual = frame[TARGET].to_numpy(dtype=float)
    table = pd.DataFrame({"target_open_utc": frame["Open_time"].to_numpy(),
                          "target_close_utc": frame["Close_time"].to_numpy(),
                          "actual_close_usdt": actual, "predicted_close_usdt": predicted,
                          "residual_usdt": actual - predicted,
                          "absolute_error_usdt": np.abs(actual - predicted)})
    return regression_metrics(actual, predicted), table


def scaler_state(scaler):
    keys = ("scale_", "min_", "data_min_", "data_max_", "data_range_")
    return {"feature_range": list(scaler.feature_range), "clip": False,
            **{key: getattr(scaler, key).tolist() for key in keys},
            "n_features_in_": int(scaler.n_features_in_), "n_samples_seen_": int(scaler.n_samples_seen_)}


def restore_scaler(state):
    scaler = MinMaxScaler(feature_range=tuple(state["feature_range"]), clip=False)
    for key in ("scale_", "min_", "data_min_", "data_max_", "data_range_"):
        setattr(scaler, key, np.asarray(state[key], dtype=float))
    scaler.n_features_in_ = int(state["n_features_in_"])
    scaler.n_samples_seen_ = int(state["n_samples_seen_"])
    return scaler


def environment_versions():
    packages = ["torch", "numpy", "pandas", "scikit-learn", "matplotlib", "seaborn", "streamlit", "requests"]
    result = {"python": platform.python_version()}
    for package in packages:
        try:
            result[package] = package_version(package)
        except Exception:
            result[package] = "not installed"
    return result


def save_bundle(root, data, fit, metrics, predictions, *, source_metadata, quality_report, experiments=None):
    model_version = "trx_1h_w25_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(root) / model_version
    run_dir.mkdir(parents=True, exist_ok=False)
    torch.save({"model_version": model_version, "state_dict": fit.state_dict}, run_dir / "weights.pt")
    write_json(run_dir / "scalers.json", {"model_version": model_version, "features": FEATURES,
               "x_scaler": scaler_state(data.x_scaler), "y_scaler": scaler_state(data.y_scaler)})
    data.frame.to_csv(run_dir / "dataset.csv", index=False)
    predictions.to_csv(run_dir / "test_predictions.csv", index=False)
    fit.history.to_csv(run_dir / "history.csv", index=False)
    data.split_table.to_csv(run_dir / "split_summary.csv", index=False)
    if experiments is not None:
        experiments.to_csv(run_dir / "experiments.csv", index=False)
    hashes = {name: file_sha256(run_dir / name) for name in ["weights.pt", "scalers.json", "dataset.csv"]}
    metadata = {"schema_version": 1, "model_version": model_version,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "symbol": SYMBOL, "interval": INTERVAL, "features": FEATURES, "target": TARGET,
                "config": asdict(fit.config), "best_epoch": fit.best_epoch,
                "training_device": fit.training_device, "training_seconds": fit.training_seconds,
                "validation_rmse_usdt": fit.validation_rmse_usdt,
                "test_metrics": metrics, "split": data.split_table.to_dict(orient="records"),
                "split_basis": "eligible target timestamps; chronological 70/15/15; integer rounding",
                "scaler_fit_last_open_utc": data.frame["Open_time"].iloc[data.scaler_fit_last_index].isoformat(),
                "source": source_metadata, "quality_report": quality_report,
                "versions": environment_versions(), "sha256": hashes,
                "evaluation_protocol": "rolling one-step; previous observed OHLCV available; no retraining"}
    write_json(run_dir / "metadata.json", metadata)
    return run_dir


def load_bundle(run_dir):
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if (metadata["schema_version"] != 1 or metadata["symbol"] != SYMBOL
            or metadata["interval"] != INTERVAL or metadata["features"] != FEATURES):
        raise ValueError("Versi/skema model tidak cocok dengan TRXUSDT 1h OHLCV.")
    config = ModelConfig(**metadata["config"])
    for name, expected in metadata["sha256"].items():
        if file_sha256(run_dir / name) != expected:
            raise ValueError(f"Artefak {name} berubah/tidak cocok; gunakan satu versi model dan scaler.")
    states = json.loads((run_dir / "scalers.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(run_dir / "weights.pt", map_location="cpu", weights_only=True)
    if (states["model_version"] != metadata["model_version"]
            or checkpoint["model_version"] != metadata["model_version"] or states["features"] != FEATURES):
        raise ValueError("Versi model dan scaler berbeda.")
    x_scaler, y_scaler = restore_scaler(states["x_scaler"]), restore_scaler(states["y_scaler"])
    if x_scaler.n_features_in_ != 5 or y_scaler.n_features_in_ != 1:
        raise ValueError("Jumlah fitur scaler tidak sesuai.")
    model = BiLSTMRegressor(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return {"model": model, "x_scaler": x_scaler, "y_scaler": y_scaler,
            "metadata": metadata, "config": config, "run_dir": run_dir}


def inference_window(frame, *, as_of=None):
    df, report = validate_frame(frame, strict=True, as_of=as_of)
    if len(df) < WINDOW_SIZE:
        raise ValueError(f"Minimal 25 candle 1h lengkap diperlukan; tersedia {len(df)}.")
    last = df.tail(WINDOW_SIZE).reset_index(drop=True)
    if not last["Open_time"].diff().dropna().eq(HOUR).all():
        raise ValueError("Ada celah/interval selain 1h pada 25 candle terakhir; prediksi dibatalkan.")
    return last, report


def predict_next(bundle, frame, *, as_of=None):
    last, report = inference_window(frame, as_of=as_of)
    x = bundle["x_scaler"].transform(last[FEATURES].to_numpy()).astype(np.float32)
    tensor = torch.from_numpy(x).unsqueeze(0)  # (1, 25, 5)
    started = perf_counter()
    with torch.inference_mode():
        scaled_prediction = bundle["model"](tensor).numpy()
    price = float(bundle["y_scaler"].inverse_transform(scaled_prediction)[0, 0])
    elapsed = perf_counter() - started
    if not np.isfinite(price):
        raise ValueError("Model menghasilkan prediksi tidak finite.")
    target_open = last["Open_time"].iloc[-1] + HOUR
    return {"predicted_close_usdt": price,
            "last_input_open_utc": last["Open_time"].iloc[-1].isoformat(),
            "last_input_close_utc": last["Close_time"].iloc[-1].isoformat(),
            "target_open_utc": target_open.isoformat(),
            "target_close_utc": (target_open + HOUR - pd.Timedelta(milliseconds=1)).isoformat(),
            "model_version": bundle["metadata"]["model_version"], "inference_seconds": elapsed,
            "features_outside_training_range": int(((x < 0) | (x > 1)).sum()),
            "quality_report": report}
