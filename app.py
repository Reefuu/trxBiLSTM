"""Jalankan: python -m streamlit run app.py (setelah notebook selesai)."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from trx_bilstm import file_sha256, inference_window, load_bundle, predict_next

ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = Path(os.environ.get("TRX_ARTIFACT_ROOT", str(ROOT / "artifacts")))
st.set_page_config(page_title="Prediksi Close TRXUSDT | BiLSTM", page_icon="📈", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1080px; padding-top: 2rem;}
.trx-title {background:#194c70;color:white;padding:1.1rem 1.4rem;border-radius:6px;
            font-size:1.35rem;font-weight:700;margin-bottom:1rem;}
[data-testid="stMetric"] {border:1px solid #9eb3c8;border-radius:7px;padding:1rem;}
.stButton button[kind="primary"] {background:#286f95; border-color:#286f95;}
</style>
<div class="trx-title">Prediksi Close TRXUSDT — BiLSTM</div>
""", unsafe_allow_html=True)
st.caption("Interval 1 jam (UTC) · 25 candle historis · target Close candle satu jam berikutnya")


@st.cache_resource(show_spinner=False)
def cached_bundle(path, metadata_hash):
    # Hash metadata berubah bila memilih hasil pelatihan baru.
    return load_bundle(path)


run_dirs = sorted([p.parent for p in ARTIFACT_ROOT.glob("trx_1h_w25_*/metadata.json")], reverse=True)
if not run_dirs:
    st.info("Model terlatih belum tersedia. Jalankan notebook TRX_BiLSTM_1h_window25.ipynb hingga tahap simpan artefak, lalu muat ulang halaman.")
    st.code("python -m streamlit run app.py", language="bash")
    st.stop()

with st.expander("Pilih model dan sumber data", expanded=True):
    chosen = st.selectbox("Versi model", options=run_dirs, format_func=lambda p: p.name)
    source = st.radio("Sumber data", ["Dataset penelitian", "Unggah CSV OHLCV"], horizontal=True)
    uploaded = st.file_uploader("CSV: Open_time, Open, High, Low, Close, Volume; Close_time opsional", type=["csv"]) if source == "Unggah CSV OHLCV" else None
    st.caption("Gunakan TRXUSDT interval 1h. Open_time ISO-8601 dengan zona waktu atau epoch ms/us didukung; waktu tanpa zona dianggap UTC.")

try:
    # Pemeriksaan hash tetap dijalankan saat rerun meskipun model tersimpan di cache.
    manifest = json.loads((chosen / "metadata.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        if file_sha256(chosen / name) != expected:
            raise ValueError(f"Artefak {name} tidak cocok dengan metadata model.")
    bundle = cached_bundle(str(chosen), file_sha256(chosen / "metadata.json"))
except Exception as exc:
    st.error(f"Model/scaler gagal dimuat: {exc}")
    st.stop()

try:
    if source == "Unggah CSV OHLCV":
        if uploaded is None:
            st.info("Unggah CSV agar validasi dan prediksi dapat dijalankan.")
            st.stop()
        raw_bytes = uploaded.getvalue()
        frame = pd.read_csv(BytesIO(raw_bytes))
    else:
        raw_bytes = (chosen / "dataset.csv").read_bytes()
        frame = pd.read_csv(BytesIO(raw_bytes))
    window, report = inference_window(frame)
except Exception as exc:
    st.error(f"Validasi gagal: {exc}")
    st.stop()

fingerprint = sha256(raw_bytes + bundle["metadata"]["model_version"].encode()).hexdigest()
if st.session_state.get("input_fingerprint") != fingerprint:
    st.session_state["input_fingerprint"] = fingerprint
    st.session_state.pop("prediction", None)

last_open = window["Open_time"].iloc[-1]
last_close = window["Close_time"].iloc[-1]
st.success(f"Data terakhir: {last_open:%Y-%m-%d %H:%M} UTC (Open time) · Status: valid, 25 candle 1h lengkap dan berurutan")
if report["unfinished_rows_removed"]:
    st.warning(f"{report['unfinished_rows_removed']} candle belum selesai dikeluarkan sebelum memilih jendela.")
if report["gap_count"]:
    st.caption(f"Dataset memiliki {report['gap_count']} celah historis; 25 candle terakhir tetap kontinu.")
st.caption(f"Candle terakhir selesai: {last_close:%Y-%m-%d %H:%M:%S} UTC. Dataset penelitian adalah snapshot saat pelatihan.")

left, right = st.columns([2.2, 1], gap="large")
with right:
    card = st.empty()
    if st.button("Jalankan Prediksi", type="primary", use_container_width=True):
        try:
            with st.spinner("Menghitung prediksi..."):
                st.session_state["prediction"] = predict_next(bundle, frame)
        except Exception as exc:
            st.session_state.pop("prediction", None)
            st.error(f"Prediksi gagal: {exc}")
    result = st.session_state.get("prediction")
    with card.container():
        if result:
            st.metric("Prediksi Close jam berikutnya", f"{result['predicted_close_usdt']:.6f} USDT")
        else:
            st.metric("Prediksi Close jam berikutnya", "—")
    if result:
        start = pd.Timestamp(result["target_open_utc"])
        end = pd.Timestamp(result["target_close_utc"])
        st.write(f"**Candle target (UTC)**\n\n{start:%Y-%m-%d %H:%M} – {end:%Y-%m-%d %H:%M:%S}")
        st.caption(f"Inferensi: {result['inference_seconds']:.4f} detik; model sudah dimuat.")
        if result["features_outside_training_range"]:
            st.caption("Sebagian fitur di luar rentang data latih; transformasi memakai scaler latih tanpa clipping.")
        if result["predicted_close_usdt"] <= 0:
            st.warning("Model regresi menghasilkan harga nonpositif. Ini keluaran model; periksa kualitas model sebelum interpretasi.")
        st.download_button("Unduh hasil prediksi", pd.DataFrame([{k: v for k, v in result.items() if k != "quality_report"}]).to_csv(index=False),
                           file_name="prediksi_TRXUSDT_1h.csv", mime="text/csv")
with left:
    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    fig.patch.set_facecolor("#f7f9fc")
    ax.set_facecolor("#f7f9fc")
    ax.plot(window["Open_time"], window["Close"], color="#2476a0", linewidth=2, label="Close historis")
    if result:
        target = pd.Timestamp(result["target_open_utc"])
        predicted = result["predicted_close_usdt"]
        ax.plot([last_open, target], [window["Close"].iloc[-1], predicted], linestyle="--", color="#d48422", linewidth=1.5)
        ax.scatter([target], [predicted], color="#d48422", label="Prediksi Close", zorder=3)
    ax.set_title("25 harga Close terakhir", loc="left", fontsize=12)
    ax.set_ylabel("Harga (USDT)")
    ax.set_xlabel("Open time candle (UTC)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Evaluasi model pada data uji")
metrics = bundle["metadata"]["test_metrics"]
c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{metrics['mae_usdt']:.6f} USDT")
c2.metric("RMSE", f"{metrics['rmse_usdt']:.6f} USDT")
c3.metric("MAPE", f"{metrics['mape_percent']:.3f}%")
test_split = next(x for x in bundle["metadata"]["split"] if x["split"] == "test")
st.caption(f"{test_split['samples']:,} sampel uji · target {test_split['first_target_utc']} hingga {test_split['last_target_utc']}. "
           "Metrik berasal dari evaluasi model tersimpan; unggahan CSV tidak menghitung ulang metrik.")
with st.expander("Keterangan model dan sumber data"):
    metadata = bundle["metadata"]
    st.write(f"Versi model: `{metadata['model_version']}`")
    st.write(f"Dilatih: {metadata['created_at_utc']} · epoch terbaik: {metadata['best_epoch']}")
    st.write("Sumber data pelatihan:", metadata["source"].get("source", "CSV penelitian"))
    st.json(metadata["config"])
    st.write("Versi perangkat lunak:")
    st.json(metadata["versions"])
st.warning("Hasil merupakan eksperimen akademik dan bukan rekomendasi transaksi maupun jaminan harga aset digital.")
