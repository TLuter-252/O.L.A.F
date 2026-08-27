from __future__ import annotations

import io
import html
import math
import zipfile
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import DualMap
from streamlit_folium import st_folium


APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
DEFAULT_DATA = APP_DIR / "data" / "ais_2023_07_04_tracks.csv.gz"
HEADER_IMAGE = REPO_DIR / "Olaf.png"
REQUIRED = {"MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG"}
PALETTE = ["#ff3b30", "#00a8ff"]

st.set_page_config(page_title="O.L.A.F. Track Outliers", page_icon="🔎", layout="wide")


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    keep = [
        c for c in [
            "MMSI", "VesselName", "BaseDateTime", "LAT", "LON", "SOG", "COG",
            "Status", "TransceiverClass", "SourceDate", "OriginalPings",
            "DurationHours", "ExtentKm", "CrossesBusiestRegion", "Briefable",
        ] if c in df
    ]
    df = df[keep].copy()
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"], errors="coerce", utc=True)
    for col in ["MMSI", "LAT", "LON", "SOG", "COG"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["OriginalPings", "DurationHours", "ExtentKm"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG"])
    df = df[df["LAT"].between(-90, 90) & df["LON"].between(-180, 180)]
    df["MMSI"] = df["MMSI"].astype("int64")
    return df.sort_values(["MMSI", "BaseDateTime"]).drop_duplicates(["MMSI", "BaseDateTime"])


@st.cache_data(show_spinner="Loading AIS tracks…")
def load_default(path: str) -> pd.DataFrame:
    return clean_frame(pd.read_csv(path, low_memory=False))


@st.cache_data(show_spinner="Reading uploaded AIS file…")
def load_upload(raw: bytes, name: str) -> pd.DataFrame:
    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            csv_names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError("The ZIP contains no CSV file.")
            with archive.open(csv_names[0]) as stream:
                return clean_frame(pd.read_csv(stream, low_memory=False))
    return clean_frame(pd.read_csv(io.BytesIO(raw), low_memory=False))


def split_segments(df: pd.DataFrame, max_gap_minutes: int) -> pd.DataFrame:
    df = df.copy()
    gaps = df.groupby("MMSI")["BaseDateTime"].diff().dt.total_seconds().div(60)
    df["segment"] = ((gaps > max_gap_minutes) | gaps.isna()).groupby(df["MMSI"]).cumsum().astype(int)
    return df


def circular_mean_deg(values: pd.Series) -> float:
    radians = np.deg2rad(values.dropna().to_numpy() % 360)
    if not len(radians):
        return np.nan
    return float(np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())) % 360)


def robust01(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    med = values.median()
    mad = (values - med).abs().median()
    if not np.isfinite(mad) or mad < 1e-9:
        ranks = values.rank(pct=True)
        return ranks.fillna(0.5)
    z = (values - med).abs() / (1.4826 * mad)
    return (1 - np.exp(-z / 2)).clip(0, 1).fillna(0)


@st.cache_data(show_spinner="Scoring route, speed, and course behavior…")
def score_vessels(df: pd.DataFrame, grid_degrees: float, min_pings: int,
                  min_duration_hours: float, min_extent_km: float,
                  route_weight: float, speed_weight: float,
                  course_weight: float) -> pd.DataFrame:
    work = df.copy()
    work["cell_lat"] = np.floor(work["LAT"] / grid_degrees).astype(int)
    work["cell_lon"] = np.floor(work["LON"] / grid_degrees).astype(int)
    occupancy = work.groupby(["cell_lat", "cell_lon"])["MMSI"].nunique()
    work = work.join(occupancy.rename("cell_vessels"), on=["cell_lat", "cell_lon"])
    work["rare_ping"] = 1 / np.sqrt(work["cell_vessels"].clip(lower=1))

    radians = np.deg2rad(work["COG"].to_numpy() % 360)
    work["course_sin"] = np.sin(radians)
    work["course_cos"] = np.cos(radians)
    grouped = work.groupby("MMSI")
    aggregations = dict(
        sampled_points=("MMSI", "size"),
        route_rarity=("rare_ping", "mean"),
        median_speed=("SOG", "median"),
        course_sin=("course_sin", "mean"),
        course_cos=("course_cos", "mean"),
        started_at=("BaseDateTime", "min"),
        ended_at=("BaseDateTime", "max"),
        min_lat=("LAT", "min"),
        max_lat=("LAT", "max"),
        min_lon=("LON", "min"),
        max_lon=("LON", "max"),
    )
    if "OriginalPings" in work:
        aggregations["original_points"] = ("OriginalPings", "max")
    if "DurationHours" in work:
        aggregations["source_duration_hours"] = ("DurationHours", "max")
    if "ExtentKm" in work:
        aggregations["source_extent_km"] = ("ExtentKm", "max")
    features = grouped.agg(**aggregations)
    features["max_speed"] = grouped["SOG"].quantile(.95)
    features["course"] = np.rad2deg(
        np.arctan2(features.pop("course_sin"), features.pop("course_cos"))
    ) % 360
    if features.empty:
        return features.reset_index()

    features["points"] = features.get("original_points", features["sampled_points"])
    observed_duration = (
        features["ended_at"] - features["started_at"]
    ).dt.total_seconds().div(3600)
    features["duration_hours"] = features.get("source_duration_hours", observed_duration)
    calculated_extent = 111.195 * np.sqrt(
        (features["max_lat"] - features["min_lat"]) ** 2
        + (
            np.cos(np.deg2rad((features["max_lat"] + features["min_lat"]) / 2))
            * (features["max_lon"] - features["min_lon"])
        ) ** 2
    )
    features["extent_km"] = features.get("source_extent_km", calculated_extent)
    features = features[
        (features["points"] >= min_pings)
        & (features["duration_hours"] >= min_duration_hours)
        & (features["extent_km"] >= min_extent_km)
    ].copy()
    if features.empty:
        return features.reset_index()

    radians = np.deg2rad(features["course"])
    center = math.atan2(np.sin(radians).mean(), np.cos(radians).mean())
    features["course_difference"] = np.abs(np.angle(np.exp(1j * (radians - center)))) / np.pi
    features["route_score"] = robust01(features["route_rarity"])
    features["speed_score"] = (robust01(features["median_speed"]) + robust01(features["max_speed"])) / 2
    features["course_score"] = robust01(features["course_difference"])
    total_weight = route_weight + speed_weight + course_weight
    features["outlier_score"] = (
        route_weight * features["route_score"]
        + speed_weight * features["speed_score"]
        + course_weight * features["course_score"]
    ) / max(total_weight, 0.001)
    features["track_score"] = (
        features["extent_km"].rank(pct=True)
        + features["duration_hours"].rank(pct=True)
    ) / 2
    # Behavioral anomaly remains dominant, but longer complete tracks rise for briefing.
    features["briefing_score"] = .80 * features["outlier_score"] + .20 * features["track_score"]
    return features.sort_values("briefing_score", ascending=False).reset_index()


def downsample(group: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if len(group) <= maximum:
        return group
    indexes = np.linspace(0, len(group) - 1, maximum, dtype=int)
    return group.iloc[indexes]


def add_tracks(map_object: folium.Map, df: pd.DataFrame, colors: dict[int, str],
               opacity: float, weight: float, max_points: int,
               tooltips: dict[int, str] | None = None) -> None:
    for (mmsi, segment), group in df.groupby(["MMSI", "segment"], sort=False):
        group = downsample(group.sort_values("BaseDateTime"), max_points)
        coords = group[["LAT", "LON"]].to_numpy().tolist()
        if len(coords) < 2:
            continue
        name = str(group["VesselName"].dropna().iloc[0]) if "VesselName" in group and group["VesselName"].notna().any() else "Unknown"
        tooltip_html = (
            tooltips.get(int(mmsi))
            if tooltips
            else f"<b>{html.escape(name)}</b><br>MMSI {mmsi}"
        )
        folium.PolyLine(
            coords, color=colors.get(int(mmsi), "#1677ff"), weight=weight,
            opacity=opacity,
            tooltip=folium.Tooltip(
                tooltip_html, sticky=True, direction="right", max_width=275
            ),
        ).add_to(map_object)


def add_baseline_tracks(map_object: folium.Map, df: pd.DataFrame) -> int:
    """Add every drawable path as one efficient GeoJSON layer."""
    lines = []
    vessel_ids = set()
    for (mmsi, _segment), group in df.groupby(["MMSI", "segment"], sort=False):
        # Keep every vessel on the map while limiting browser-side geometry.
        group = downsample(group, 8)
        coordinates = group[["LON", "LAT"]].to_numpy().tolist()
        if len(coordinates) >= 2:
            vessel_ids.add(int(mmsi))
            lines.append(coordinates)
    folium.GeoJson(
        {
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "MultiLineString", "coordinates": lines},
        },
        style_function=lambda _feature: {
            "color": "#006dff", "weight": 2.25, "opacity": 0.62,
        },
        name="Baseline vessel tracks",
    ).add_to(map_object)
    return len(vessel_ids)


@st.cache_data(show_spinner=False)
def densest_traffic_region(df: pd.DataFrame, cell_size: float = 0.5) -> dict:
    """Find the regional cell crossed by the greatest number of unique vessels."""
    cells = df[["MMSI", "LAT", "LON"]].copy()
    cells["lat_cell"] = np.floor(cells["LAT"] / cell_size).astype(int)
    cells["lon_cell"] = np.floor(cells["LON"] / cell_size).astype(int)
    traffic = cells.groupby(["lat_cell", "lon_cell"])["MMSI"].nunique()
    if traffic.empty:
        return {"center": [27.8, -82.5], "bounds": [27.55, 28.05, -82.75, -82.25]}
    lat_cell, lon_cell = traffic.idxmax()
    busiest = cells[(cells["lat_cell"] == lat_cell) & (cells["lon_cell"] == lon_cell)]
    return {
        "center": [float(busiest["LAT"].median()), float(busiest["LON"].median())],
        "bounds": [
            lat_cell * cell_size, (lat_cell + 1) * cell_size,
            lon_cell * cell_size, (lon_cell + 1) * cell_size,
        ],
    }


def synchronized_map(baseline: pd.DataFrame, outliers: pd.DataFrame,
                     highlighted: list[int], center: list[float],
                     outlier_tooltips: dict[int, str]) -> DualMap:
    """Build two Leaflet maps whose center and zoom always stay synchronized."""
    result = DualMap(location=center, zoom_start=8, tiles=None, control_scale=True)

    # OpenStreetMap's standard tiles do not need an API key.
    for pane in (result.m1, result.m2):
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=False).add_to(pane)

    add_baseline_tracks(result.m1, baseline)
    colors = {mmsi: PALETTE[i % len(PALETTE)] for i, mmsi in enumerate(highlighted)}
    add_tracks(result.m2, outliers, colors, .95, 5, 800, outlier_tooltips)

    return result


def build_outlier_tooltips(scores: pd.DataFrame, selected: list[int],
                           weights: dict[str, float]) -> dict[int, str]:
    """Explain each selected vessel using the analyst's current weighting."""
    explanations = {
        "route": "Route uses traffic cells visited by relatively few peer vessels.",
        "speed": "Median or peak speed differs from the peer-vessel pattern.",
        "course": "Mean course differs from the dominant local traffic flow.",
    }
    labels = {"route": "Route rarity", "speed": "Speed", "course": "Course"}
    result = {}
    total_weight = max(sum(weights.values()), 0.001)
    indexed = scores.set_index("MMSI")
    for mmsi in selected:
        row = indexed.loc[mmsi]
        components = []
        for key in ["route", "speed", "course"]:
            score = float(row[f"{key}_score"])
            contribution = weights[key] * score / total_weight
            components.append((key, score, contribution))
        components.sort(key=lambda item: item[2], reverse=True)
        primary = components[0]
        secondary = components[1]
        reason = explanations[primary[0]]
        if secondary[2] >= primary[2] * 0.75 and weights[secondary[0]] > 0:
            reason += f" {labels[secondary[0]]} was also a major contributor."
        result[int(mmsi)] = (
            f"<div style='width:250px;line-height:1.35;white-space:normal'>"
            f"<b>MMSI {int(mmsi)}</b><br>"
            f"<b>Why flagged:</b> {reason}<br>"
            f"<b>Briefing rank:</b> {float(row['briefing_score']) * 100:.1f}%<br>"
            f"<b>Overall outlier score:</b> {float(row['outlier_score']) * 100:.1f}%<br>"
            f"Route {float(row['route_score']) * 100:.0f}% &nbsp;·&nbsp; "
            f"Speed {float(row['speed_score']) * 100:.0f}% &nbsp;·&nbsp; "
            f"Course {float(row['course_score']) * 100:.0f}%<br>"
            f"{int(row['points'])} pings &nbsp;·&nbsp; "
            f"{float(row['duration_hours']):.1f} hours &nbsp;·&nbsp; "
            f"{float(row['extent_km']):,.0f} km extent<br>"
            f"median SOG {float(row['median_speed']):.1f} kn &nbsp;·&nbsp; "
            f"95th percentile SOG {float(row['max_speed']):.1f} kn"
            f"</div>"
        )
    return result


with st.sidebar:
    st.header("Analyst controls")
    uploaded = st.file_uploader("AIS CSV or ZIP", type=["csv", "zip"], help="Leave empty to use the cloud-hosted July 4, 2023 demo data.")
    min_speed = st.slider("Outlier minimum speed (knots)", 0.0, 30.0, 0.5, 0.5)
    max_gap = st.slider("Break track after gap (minutes)", 15, 720, 240, 15)
    min_pings = st.slider("Minimum original pings", 20, 500, 40, 10)
    st.caption("Right-map tracks must cover at least 6 hours and 30 km.")
    grid_size = st.slider("Traffic grid size (degrees)", 0.01, 0.25, 0.05, 0.01)
    st.caption("Choose what 'unusual' means")
    route_weight = st.slider("Route / traffic pattern", 0.0, 3.0, 2.0, 0.25)
    speed_weight = st.slider("Speed behavior", 0.0, 3.0, 1.0, 0.25)
    course_weight = st.slider("Course behavior", 0.0, 3.0, 1.0, 0.25)
    result_count = st.radio("Tracks on right map", [1, 2], horizontal=True, index=1)

head, art = st.columns([4, 1])
with head:
    st.title("O.L.A.F.")
    st.subheader("Outlier & Low-frequency Analysis Framework")
    st.caption("All vessel tracks on the left. Only the strongest analyst-tuned outliers on the right.")
with art:
    if HEADER_IMAGE.exists():
        st.image(str(HEADER_IMAGE), use_container_width=True)

try:
    ais = load_upload(uploaded.getvalue(), uploaded.name) if uploaded else load_default(str(DEFAULT_DATA))
except Exception as exc:
    st.error(f"AIS data could not be loaded: {exc}")
    st.stop()

if not uploaded and "SourceDate" in ais:
    st.info(
        "Built-in briefing data: July 4, 2023 — the strongest tested day for total "
        "vessel traffic, long tracks, and useful tracks in the busiest region."
    )

baseline = split_segments(ais, max_gap)
filtered = baseline[baseline["SOG"] >= min_speed].copy()
busy_region = densest_traffic_region(baseline)
lat_min, lat_max, lon_min, lon_max = busy_region["bounds"]
region_rows = filtered[
    filtered["LAT"].between(lat_min, lat_max)
    & filtered["LON"].between(lon_min, lon_max)
]
if "CrossesBusiestRegion" in filtered:
    region_mmsi = filtered.loc[filtered["CrossesBusiestRegion"].astype(bool), "MMSI"].unique()
else:
    region_mmsi = region_rows["MMSI"].unique()
scoring_tracks = filtered[filtered["MMSI"].isin(region_mmsi)].copy()
display_baseline = baseline[baseline["MMSI"].isin(region_mmsi)].copy()
# The left side is a full-day overview, so show one continuous path per vessel.
display_baseline["segment"] = 1
scores = score_vessels(
    scoring_tracks, grid_size, min_pings, 6.0, 30.0,
    route_weight, speed_weight, course_weight,
)
if scores.empty:
    st.warning("No vessels meet the current filters. Lower Minimum pings or Minimum speed.")
    st.stop()

selected = scores.head(result_count)["MMSI"].astype(int).tolist()
outlier_tracks = filtered[filtered["MMSI"].isin(selected)].copy()
outlier_tooltips = build_outlier_tooltips(
    scores,
    selected,
    {"route": route_weight, "speed": speed_weight, "course": course_weight},
)

left, right = st.columns(2)
with left:
    segment_sizes = display_baseline.groupby(["MMSI", "segment"]).size()
    shown_count = segment_sizes[segment_sizes >= 2].index.get_level_values("MMSI").nunique()
    st.subheader(f"Baseline · {shown_count:,} complete tracks in busiest region")
with right:
    st.subheader(f"Tracks of interest · top {len(selected)}")
st.caption("The maps are linked: pan or zoom either side and the other side follows.")
st.caption("Outliers are ranked among vessels crossing the busiest default region.")
st_folium(
    synchronized_map(
        display_baseline, outlier_tracks, selected, busy_region["center"], outlier_tooltips
    ),
    height=620,
    use_container_width=True,
    returned_objects=[],
)

st.subheader("Why these tracks were flagged")
display = scores.head(result_count).copy()
display["MMSI"] = display["MMSI"].astype(str)
for col in ["briefing_score", "outlier_score", "route_score", "speed_score", "course_score"]:
    display[col] = (display[col] * 100).round(1)
st.dataframe(
    display[[
        "MMSI", "points", "duration_hours", "extent_km", "briefing_score",
        "outlier_score", "route_score", "speed_score", "course_score",
        "median_speed", "max_speed",
    ]],
    column_config={
        "duration_hours": "Hours", "extent_km": "Extent km",
        "briefing_score": "Briefing rank %", "outlier_score": "Outlier %",
        "route_score": "Route %", "speed_score": "Speed %",
        "course_score": "Course %", "median_speed": "Median SOG",
        "max_speed": "95th % SOG",
    }, hide_index=True, use_container_width=True,
)

choice = st.selectbox("Selected MMSI", [str(m) for m in selected])
st.link_button("Identify vessel on MarineTraffic", f"https://www.marinetraffic.com/en/ais/details/ships/mmsi:{choice}", use_container_width=True)
export = outlier_tracks[outlier_tracks["MMSI"] == int(choice)].drop(columns="segment", errors="ignore")
st.download_button("Download selected track", export.to_csv(index=False), f"OLAF_MMSI_{choice}.csv", "text/csv", use_container_width=True)
