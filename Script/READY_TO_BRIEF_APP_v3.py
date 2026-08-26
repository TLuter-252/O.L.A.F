import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import folium
import streamlit.components.v1 as components

# Optional: better folium support if installed
try:
    from streamlit_folium import st_folium
except Exception:
    st_folium = None


# ----------------------------
# SETTINGS (keep these simple)
# ----------------------------
DEFAULT_AIS_PATH = r"C:\Users\1381358760.MIL\OneDrive - US Army\Desktop\AIS Data Cleaned"
# HEADER_IMAGE_PATH = r"C:\CODE\A_Florida\data\Olaf.png"

AO_LAT_MIN, AO_LAT_MAX = 25.0, 30.0
AO_LON_MIN, AO_LON_MAX = -85.0, -80.0

USE_COLS = ["MMSI", "VesselName", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Status", "TransceiverClass"]

BASELINE_SOG_MIN = 0.0
BASELINE_TRAIN_FRACTION = 0.80
BASELINE_MAX_TOTAL_POINTS = 120_000
BASELINE_TARGET_POINTS_PER_MMSI = 140
BASELINE_MIN_POINTS_PER_MMSI = 25

HEATMAP_CAP = 250_000
RIGHT_POINTS_PER_VESSEL = 250
RIGHT_MAX_TOTAL_POINTS = 70_000
RIGHT_MAX_VESSELS = 60

# Cache files (simple + works on any OS)
PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PARQUET = DATA_DIR / "Florida_Routes.filtered.parquet"
CACHE_META = DATA_DIR / "Florida_Routes.filtered.meta.txt"


# ----------------------------
# UI HEADER
# ----------------------------
st.set_page_config(page_title="O.L.A.F.", layout="wide")

h1, h2 = st.columns([3, 1], vertical_alignment="center")
with h1:
    st.markdown(
        """
        <div style="line-height:1.05;">
          <div style="font-size: 2.25rem; font-weight: 800;">O.L.A.F.</div>
          <div style="font-size: 1.45rem; font-weight: 600; opacity: 0.85;">
            Outlier &amp; Low-frequency Analysis Framework
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Baseline routes on the left. Outlier tracks on the right.")
with h2:
    if os.path.exists(HEADER_IMAGE_PATH):
        st.image(HEADER_IMAGE_PATH, use_container_width=True)

st.session_state.setdefault("clicked_mmsi", "")
st.session_state.setdefault("ais_class_mode", "Class B")


# ----------------------------
# SMALL HELPERS
# ----------------------------
def show_map(m, height=650):
    """Render folium map in Streamlit (best method if st_folium is installed)."""
    if st_folium:
        return st_folium(m, height=height, width=None)
    components.html(m.get_root().render(), height=height, scrolling=True)
    return {}


def csv_stamp(path: str) -> str:
    """A quick signature so we know if the CSV changed."""
    try:
        s = os.stat(path)
        return f"{os.path.abspath(path)}|{int(s.st_mtime)}|{int(s.st_size)}"
    except Exception:
        return ""


def read_meta() -> str:
    try:
        return CACHE_META.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def write_meta(txt: str):
    try:
        CACHE_META.write_text(txt, encoding="utf-8")
    except Exception:
        pass


def filter_class(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if df.empty or "TransceiverClass" not in df.columns:
        return df
    if mode == "Class A":
        return df[df["TransceiverClass"] == "A"].copy()
    if mode == "Class B":
        return df[df["TransceiverClass"] == "B"].copy()
    return df.copy()


def remove_time_jumps(df: pd.DataFrame, max_gap_min: int) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(["MMSI", "BaseDateTime"])
    gap = df.groupby("MMSI")["BaseDateTime"].diff().dt.total_seconds() / 60
    return df[gap.isna() | (gap <= max_gap_min)].copy()


def split_by_time(df: pd.DataFrame, frac: float):
    if df.empty:
        return df, df
    df = df.sort_values("BaseDateTime")
    cut = int(len(df) * frac)
    cut = max(1, min(cut, len(df) - 1))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def downsample(df: pd.DataFrame, target_points: int) -> pd.DataFrame:
    if len(df) <= target_points:
        return df
    step = max(1, int(np.ceil(len(df) / target_points)))
    return df.iloc[::step].copy()


def color_for(mmsi: int) -> str:
    colors = [
        "#FF0000", "#00BFFF", "#00FF00", "#FF00FF", "#FFA500", "#00FFFF",
        "#FFD700", "#7FFF00", "#FF1493", "#1E90FF", "#00FF7F", "#FF4500",
        "#8A2BE2", "#ADFF2F", "#FF69B4", "#00CED1", "#FFFF00", "#FF6347",
    ]
    return colors[int(mmsi) % len(colors)]


# ----------------------------
# LOAD + CLEAN DATA (simple cache)
# ----------------------------
def build_filtered_df_from_csv(csv_path: str) -> pd.DataFrame:
    chunks = []
    for ch in pd.read_csv(csv_path, usecols=USE_COLS, chunksize=500_000, low_memory=True):
        ch.columns = ch.columns.str.strip()

        ch["BaseDateTime"] = pd.to_datetime(ch["BaseDateTime"], errors="coerce")
        for c in ["LAT", "LON", "SOG", "COG"]:
            ch[c] = pd.to_numeric(ch[c], errors="coerce")

        ch = ch[
            ch["LAT"].between(AO_LAT_MIN, AO_LAT_MAX)
            & ch["LON"].between(AO_LON_MIN, AO_LON_MAX)
        ]
        ch = ch.dropna(subset=["MMSI", "BaseDateTime", "LAT", "LON", "SOG"])
        if not ch.empty:
            chunks.append(ch)

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)

    df["MMSI"] = pd.to_numeric(df["MMSI"], errors="coerce")
    df = df.dropna(subset=["MMSI"])
    df["MMSI"] = df["MMSI"].astype(int)

    if "TransceiverClass" in df.columns:
        df["TransceiverClass"] = df["TransceiverClass"].astype(str).str.upper().str.strip()

    # Save a parquet cache (fast)
    try:
        df.to_parquet(CACHE_PARQUET, index=False)
    except Exception:
        pass

    return df


@st.cache_data(show_spinner=True)
def load_ais(csv_path: str) -> pd.DataFrame:
    if not csv_path or not os.path.exists(csv_path):
        return pd.DataFrame()

    stamp_now = csv_stamp(csv_path)
    stamp_old = read_meta()

    # If CSV has not changed, use cached parquet if possible
    if stamp_now == stamp_old and CACHE_PARQUET.exists():
        try:
            return pd.read_parquet(CACHE_PARQUET)
        except Exception:
            pass

    df = build_filtered_df_from_csv(csv_path)
    write_meta(stamp_now)
    return df


# ----------------------------
# SCORING (baseline grid + low-traffic %)
# ----------------------------
@st.cache_data(show_spinner=False)
def make_baseline_grid(baseline_train: pd.DataFrame, grid_size: float) -> pd.DataFrame:
    if baseline_train.empty:
        return pd.DataFrame(columns=["lat_cell", "lon_cell", "count"])

    df = baseline_train
    if len(df) > HEATMAP_CAP:
        df = df.sample(HEATMAP_CAP, random_state=42)

    tmp = df[["LAT", "LON"]].copy()
    tmp["lat_cell"] = (tmp["LAT"] / grid_size).round() * grid_size
    tmp["lon_cell"] = (tmp["LON"] / grid_size).round() * grid_size
    return tmp.groupby(["lat_cell", "lon_cell"]).size().reset_index(name="count")


@st.cache_data(show_spinner=False)
def score_everyone(user_test: pd.DataFrame, baseline_grid: pd.DataFrame, grid_size: float, low_pct: float) -> pd.DataFrame:
    if user_test.empty or baseline_grid.empty:
        return pd.DataFrame(columns=["MMSI", "low_traffic_percent", "points"])

    threshold = baseline_grid["count"].quantile(low_pct)

    tmp = user_test[["MMSI", "LAT", "LON"]].copy()
    tmp["lat_cell"] = (tmp["LAT"] / grid_size).round() * grid_size
    tmp["lon_cell"] = (tmp["LON"] / grid_size).round() * grid_size

    bg = baseline_grid.set_index(["lat_cell", "lon_cell"])
    tmp = tmp.join(bg, on=["lat_cell", "lon_cell"])
    tmp["count"] = tmp["count"].fillna(0)
    tmp["low"] = tmp["count"] <= threshold

    return (
        tmp.groupby("MMSI", as_index=False)
        .agg(low_traffic_percent=("low", "mean"), points=("low", "size"))
        .sort_values("low_traffic_percent", ascending=False)
    )


# ----------------------------
# MAP BUILDERS
# ----------------------------
def build_left_map(baseline_train: pd.DataFrame, max_gap_min: int) -> folium.Map:
    if baseline_train.empty:
        return folium.Map(location=[27.9, -82.4], zoom_start=10, tiles="CartoDB positron")

    m = folium.Map(
        location=[float(baseline_train["LAT"].median()), float(baseline_train["LON"].median())],
        zoom_start=10,
        tiles="CartoDB positron",
    )

    total = 0
    base_color = "#2b6cb0"
    order = baseline_train.groupby("MMSI").size().sort_values(ascending=False).index.tolist()

    for mmsi in order:
        g = baseline_train[baseline_train["MMSI"] == mmsi].sort_values("BaseDateTime")
        if len(g) < BASELINE_MIN_POINTS_PER_MMSI:
            continue

        g = downsample(g, BASELINE_TARGET_POINTS_PER_MMSI)

        chunk, last_t = [], None
        for row in g.itertuples(index=False):
            t = row.BaseDateTime
            pt = (float(row.LAT), float(row.LON))

            if last_t is not None and (t - last_t).total_seconds() / 60 > max_gap_min:
                if len(chunk) >= 2:
                    folium.PolyLine(chunk, color=base_color, weight=3, opacity=0.20).add_to(m)
                    total += len(chunk)
                chunk = []

            chunk.append(pt)
            last_t = t

        if len(chunk) >= 2:
            folium.PolyLine(chunk, color=base_color, weight=3, opacity=0.20).add_to(m)
            total += len(chunk)

        if total >= BASELINE_MAX_TOTAL_POINTS:
            break

    return m


@st.cache_data(show_spinner=False)
def left_map_html_cached(baseline_train: pd.DataFrame, max_gap_min: int) -> str:
    return build_left_map(baseline_train, max_gap_min).get_root().render()


def build_right_map(tracks: pd.DataFrame, max_gap_min: int) -> folium.Map:
    if tracks.empty:
        return folium.Map(location=[27.9, -82.4], zoom_start=10, tiles="CartoDB positron")

    m = folium.Map(
        location=[float(tracks["LAT"].median()), float(tracks["LON"].median())],
        zoom_start=10,
        tiles="CartoDB positron",
    )

    total = 0
    order = (
        tracks.groupby("MMSI").size().sort_values(ascending=False).index.tolist()[:RIGHT_MAX_VESSELS]
    )

    for mmsi in order:
        g = tracks[tracks["MMSI"] == mmsi].sort_values("BaseDateTime")
        g = downsample(g, RIGHT_POINTS_PER_VESSEL)

        color = color_for(mmsi)
        chunk, last_t = [], None

        for row in g.itertuples(index=False):
            t = row.BaseDateTime
            pt = (float(row.LAT), float(row.LON))

            if last_t is not None and (t - last_t).total_seconds() / 60 > max_gap_min:
                if len(chunk) >= 2:
                    folium.PolyLine(chunk, color=color, weight=4, opacity=0.85,
                                    tooltip=f"MMSI: {mmsi}", popup=f"MMSI: {mmsi}").add_to(m)
                    total += len(chunk)
                chunk = []

            chunk.append(pt)
            last_t = t

        if len(chunk) >= 2:
            folium.PolyLine(chunk, color=color, weight=4, opacity=0.85,
                            tooltip=f"MMSI: {mmsi}", popup=f"MMSI: {mmsi}").add_to(m)
            total += len(chunk)

        if total >= RIGHT_MAX_TOTAL_POINTS:
            break

    return m


# ----------------------------
# SIDEBAR CONTROLS
# ----------------------------
st.sidebar.markdown("## AIS Filters")

AIS_PATH = st.sidebar.text_input("AIS CSV path", DEFAULT_AIS_PATH)

st.session_state["ais_class_mode"] = st.sidebar.selectbox(
    "AIS Class (RIGHT map only)",
    ["All", "Class A", "Class B"],
    index=["All", "Class A", "Class B"].index(st.session_state["ais_class_mode"]),
)

sog_min = st.sidebar.number_input("SOG min", 0.0, 20.0, 8.00)
max_gap_min = st.sidebar.number_input("Max time gap (minutes)", 1, 240, 5)
train_frac = st.sidebar.number_input("Train fraction", 0.5, 0.95, 0.80)
grid_size = st.sidebar.number_input("Grid size (scoring)", 0.0005, 0.02, 0.0012, step=0.0001, format="%.4f")
low_pct = st.sidebar.number_input("Low-traffic percentile", 0.01, 0.3, 0.08)
min_pts = st.sidebar.number_input("Min pings per vessel", 10, 10000, 350)
top_k = st.sidebar.number_input("Max variables", 1, 400, 5)
show_only_topk = st.sidebar.checkbox("Right map: show ONLY Top choices", value=True)


# ----------------------------
# LOAD DATA
# ----------------------------
ais = load_ais(AIS_PATH)
if ais.empty:
    st.error("No AIS data loaded. Check the CSV path / columns.")
    st.stop()


# ----------------------------
# PIPELINE
# ----------------------------
baseline_tracks = ais[ais["SOG"] >= BASELINE_SOG_MIN].copy()
baseline_tracks = remove_time_jumps(baseline_tracks, int(max_gap_min))
baseline_train, _ = split_by_time(baseline_tracks, float(BASELINE_TRAIN_FRACTION))

user_df = filter_class(ais, st.session_state["ais_class_mode"])
user_df = user_df[user_df["SOG"] >= float(sog_min)].copy()
user_df = remove_time_jumps(user_df, int(max_gap_min))
_, user_test = split_by_time(user_df, float(train_frac))

baseline_grid = make_baseline_grid(baseline_train, float(grid_size))
scores_all = score_everyone(user_test, baseline_grid, float(grid_size), float(low_pct))

scores = scores_all[scores_all["points"] >= int(min_pts)].copy()
sus = scores.head(int(top_k))["MMSI"].tolist() if not scores.empty else []

tracks_show = user_test[user_test["MMSI"].isin(sus)].copy() if show_only_topk else user_test.copy()


# ----------------------------
# MAIN LAYOUT
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Baseline")
    st.markdown("<div style='height: 85px;'></div>", unsafe_allow_html=True)
    html = left_map_html_cached(baseline_train, int(max_gap_min))
    components.html(html, height=650, scrolling=True)

with col2:
    st.subheader("Tracks of Interest")

    mmsi_list = sorted(tracks_show["MMSI"].unique().tolist()) if not tracks_show.empty else []
    chosen = st.selectbox("Focus MMSI (RIGHT map)", ["All (Top)"] + [str(x) for x in mmsi_list], index=0)

    tracks_right = tracks_show
    if chosen != "All (Top)":
        tracks_right = tracks_show[tracks_show["MMSI"] == int(chosen)]

    state = show_map(build_right_map(tracks_right, int(max_gap_min)))

    if isinstance(state, dict):
        combined = str(state.get("last_object_clicked_tooltip", "")) + str(state.get("last_object_clicked_popup", ""))
        m = re.search(r"MMSI\s*:\s*(\d+)", combined)
        if m:
            st.session_state["clicked_mmsi"] = m.group(1)


# ----------------------------
# CLICK MMSI + IDENTIFY BUTTON
# ----------------------------
st.markdown("---")
st.subheader("Selected MMSI")

if st.session_state["clicked_mmsi"]:
    mmsi = st.session_state["clicked_mmsi"]
    st.write(f"**Clicked MMSI:** `{mmsi}`")
    components.html(f"<script>navigator.clipboard.writeText('{mmsi}');</script>", height=0)
    st.caption("Copied to clipboard. Now click Identify.")
else:
    st.caption("Click a track on the RIGHT map to copy its MMSI, then click Identify.")

st.markdown(
    """
    <a href="https://www.marinetraffic.com/en/ais/details/ships" target="_blank" style="text-decoration:none;">
      <div style="
        margin-top:10px;
        padding:18px;
        border-radius:12px;
        background:#0e3a5b;
        color:white;
        font-size:20px;
        font-weight:700;
        text-align:center;
        cursor:pointer;">
        Identify
      </div>
    </a>
    """,
    unsafe_allow_html=True,
)


# ----------------------------
# DOWNLOAD CSV (RIGHT MAP)
# ----------------------------
EXPORT_COLS = ["MMSI", "VesselName", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Status", "TransceiverClass"]

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

if isinstance(locals().get("tracks_right"), pd.DataFrame) and not tracks_right.empty:
    export_df = tracks_right.copy()

    for c in EXPORT_COLS:
        if c not in export_df.columns:
            export_df[c] = ""

    export_df = export_df[EXPORT_COLS].copy()
    export_df["BaseDateTime"] = pd.to_datetime(export_df["BaseDateTime"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
