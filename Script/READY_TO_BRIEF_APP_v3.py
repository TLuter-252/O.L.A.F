import re
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    from streamlit_folium import st_folium
except Exception:
    st_folium = None


# ----------------------------
# SETTINGS
# ----------------------------
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/TLuter-252/O.L.A.F/main"
AIS_DATA_URL = f"{GITHUB_RAW_BASE}/Florida_routes.csv"
HEADER_IMAGE_URL = f"{GITHUB_RAW_BASE}/Olaf.png"

AO_LAT_MIN, AO_LAT_MAX = 25.0, 30.0
AO_LON_MIN, AO_LON_MAX = -85.0, -80.0

USE_COLS = [
    "MMSI", "VesselName", "BaseDateTime", "LAT", "LON",
    "SOG", "COG", "Status", "TransceiverClass",
]

BASELINE_SOG_MIN = 0.0
BASELINE_TRAIN_FRACTION = 0.80
BASELINE_MAX_TOTAL_POINTS = 120_000
BASELINE_TARGET_POINTS_PER_MMSI = 140
BASELINE_MAX_VESSELS = 500

HEATMAP_CAP = 250_000
RIGHT_POINTS_PER_VESSEL = 250
RIGHT_MAX_TOTAL_POINTS = 70_000
RIGHT_MAX_VESSELS = 60

PROJECT = Path(__file__).resolve().parent
DATA_DIR = PROJECT / ".olaf_cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PARQUET = DATA_DIR / "Florida_Routes.filtered.parquet"


# ----------------------------
# PAGE HEADER
# ----------------------------
st.set_page_config(
    page_title="O.L.A.F.",
    layout="wide",
)

h1, h2 = st.columns(
    [3, 1],
    vertical_alignment="center",
)

with h1:
    st.markdown(
        """
        <div style="line-height:1.05;">
          <div style="font-size:2.25rem;font-weight:800;">
            O.L.A.F.
          </div>

          <div style="font-size:1.45rem;font-weight:600;opacity:0.85;">
            Outlier &amp; Low-frequency Analysis Framework
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Baseline routes on the left. "
        "Outlier tracks on the right."
    )

with h2:
    st.image(
        HEADER_IMAGE_URL,
        use_container_width=True,
    )


st.session_state.setdefault(
    "clicked_mmsi",
    "",
)

st.session_state.setdefault(
    "ais_class_mode",
    "All",
)


# ----------------------------
# HELPERS
# ----------------------------
def show_map(m, height=650):
    """
    Render a Folium map in Streamlit.
    """

    if st_folium:
        return st_folium(
            m,
            height=height,
            width=None,
        )

    components.html(
        m.get_root().render(),
        height=height,
        scrolling=True,
    )

    return {}


def filter_class(df, mode):
    """
    Filter AIS records by transceiver class.
    """

    if (
        df.empty
        or "TransceiverClass" not in df.columns
    ):
        return df.copy()

    if mode == "Class A":
        return df[
            df["TransceiverClass"] == "A"
        ].copy()

    if mode == "Class B":
        return df[
            df["TransceiverClass"] == "B"
        ].copy()

    return df.copy()


def split_by_time(df, frac):
    """
    Split data into an earlier training period
    and a later testing period.
    """

    if df.empty:
        return (
            df.copy(),
            df.copy(),
        )

    if len(df) == 1:
        return (
            df.copy(),
            df.copy(),
        )

    df = (
        df.sort_values("BaseDateTime")
        .reset_index(drop=True)
    )

    cut = int(
        len(df) * frac
    )

    cut = max(
        1,
        min(
            cut,
            len(df) - 1,
        ),
    )

    return (
        df.iloc[:cut].copy(),
        df.iloc[cut:].copy(),
    )


def downsample(
    df,
    target_points,
):
    """
    Reduce very large tracks before drawing them.
    """

    if len(df) <= target_points:
        return df

    step = max(
        1,
        int(
            np.ceil(
                len(df)
                / target_points
            )
        ),
    )

    return df.iloc[::step].copy()


def color_for(mmsi):
    """
    Give each vessel a repeatable color.
    """

    colors = [
        "#FF0000",
        "#00BFFF",
        "#00FF00",
        "#FF00FF",
        "#FFA500",
        "#00FFFF",
        "#FFD700",
        "#7FFF00",
        "#FF1493",
        "#1E90FF",
        "#00FF7F",
        "#FF4500",
        "#8A2BE2",
        "#ADFF2F",
        "#FF69B4",
        "#00CED1",
        "#FFFF00",
        "#FF6347",
    ]

    return colors[
        int(mmsi)
        % len(colors)
    ]


def fit_map_to_data(
    m,
    df,
):
    """
    Automatically zoom the map to the AIS data.
    """

    if df.empty:
        return

    lat_min = float(
        df["LAT"].min()
    )

    lat_max = float(
        df["LAT"].max()
    )

    lon_min = float(
        df["LON"].min()
    )

    lon_max = float(
        df["LON"].max()
    )

    values = np.array(
        [
            lat_min,
            lat_max,
            lon_min,
            lon_max,
        ],
        dtype=float,
    )

    if not np.isfinite(values).all():
        return

    if lat_min == lat_max:
        lat_min -= 0.01
        lat_max += 0.01

    if lon_min == lon_max:
        lon_min -= 0.01
        lon_max += 0.01

    m.fit_bounds(
        [
            [
                lat_min,
                lon_min,
            ],
            [
                lat_max,
                lon_max,
            ],
        ]
    )


# ----------------------------
# LOAD + CLEAN AIS DATA
# ----------------------------
def build_filtered_df_from_csv(
    csv_path,
):
    """
    Read the AIS CSV from GitHub and clean
    the values needed by O.L.A.F.
    """

    chunks = []

    for ch in pd.read_csv(
        csv_path,
        usecols=USE_COLS,
        chunksize=500_000,
        low_memory=True,
    ):

        ch.columns = (
            ch.columns
            .str.strip()
        )

        ch[
            "BaseDateTime"
        ] = pd.to_datetime(
            ch["BaseDateTime"],
            errors="coerce",
        )

        ch[
            "MMSI"
        ] = pd.to_numeric(
            ch["MMSI"],
            errors="coerce",
        )

        for col in [
            "LAT",
            "LON",
            "SOG",
            "COG",
        ]:

            ch[
                col
            ] = pd.to_numeric(
                ch[col],
                errors="coerce",
            )

        # Florida operating area.
        ch = ch[
            ch["LAT"].between(
                AO_LAT_MIN,
                AO_LAT_MAX,
            )
            &
            ch["LON"].between(
                AO_LON_MIN,
                AO_LON_MAX,
            )
        ]

        ch = ch.dropna(
            subset=[
                "MMSI",
                "BaseDateTime",
                "LAT",
                "LON",
                "SOG",
            ]
        )

        if not ch.empty:
            chunks.append(ch)

    if not chunks:
        return pd.DataFrame(
            columns=USE_COLS
        )

    df = pd.concat(
        chunks,
        ignore_index=True,
    )

    df[
        "MMSI"
    ] = df[
        "MMSI"
    ].astype(
        "int64"
    )

    if (
        "TransceiverClass"
        in df.columns
    ):

        df[
            "TransceiverClass"
        ] = (
            df[
                "TransceiverClass"
            ]
            .astype(str)
            .str.upper()
            .str.strip()
        )

    try:
        df.to_parquet(
            CACHE_PARQUET,
            index=False,
        )

    except Exception:
        pass

    return df


@st.cache_data(
    show_spinner="Loading AIS data...",
    ttl=3600,
)
def load_ais(
    csv_source,
):

    return (
        build_filtered_df_from_csv(
            csv_source
        )
    )


# ----------------------------
# OUTLIER SCORING
# ----------------------------
@st.cache_data(
    show_spinner=False
)
def make_baseline_grid(
    baseline_train,
    grid_size,
):

    if baseline_train.empty:

        return pd.DataFrame(
            columns=[
                "lat_cell",
                "lon_cell",
                "count",
            ]
        )

    df = baseline_train

    if len(df) > HEATMAP_CAP:

        df = df.sample(
            HEATMAP_CAP,
            random_state=42,
        )

    tmp = df[
        [
            "LAT",
            "LON",
        ]
    ].copy()

    tmp[
        "lat_cell"
    ] = (
        (
            tmp["LAT"]
            / grid_size
        ).round()
        * grid_size
    )

    tmp[
        "lon_cell"
    ] = (
        (
            tmp["LON"]
            / grid_size
        ).round()
        * grid_size
    )

    return (
        tmp.groupby(
            [
                "lat_cell",
                "lon_cell",
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )


@st.cache_data(
    show_spinner=False
)
def score_everyone(
    user_test,
    baseline_grid,
    grid_size,
    low_pct,
):

    if (
        user_test.empty
        or baseline_grid.empty
    ):

        return pd.DataFrame(
            columns=[
                "MMSI",
                "low_traffic_percent",
                "points",
            ]
        )

    threshold = (
        baseline_grid[
            "count"
        ]
        .quantile(
            low_pct
        )
    )

    tmp = user_test[
        [
            "MMSI",
            "LAT",
            "LON",
        ]
    ].copy()

    tmp[
        "lat_cell"
    ] = (
        (
            tmp["LAT"]
            / grid_size
        ).round()
        * grid_size
    )

    tmp[
        "lon_cell"
    ] = (
        (
            tmp["LON"]
            / grid_size
        ).round()
        * grid_size
    )

    lookup = (
        baseline_grid
        .set_index(
            [
                "lat_cell",
                "lon_cell",
            ]
        )
    )

    tmp = tmp.join(
        lookup,
        on=[
            "lat_cell",
            "lon_cell",
        ],
    )

    tmp[
        "count"
    ] = (
        tmp[
            "count"
        ]
        .fillna(0)
    )

    tmp[
        "low"
    ] = (
        tmp[
            "count"
        ]
        <= threshold
    )

    return (
        tmp.groupby(
            "MMSI",
            as_index=False,
        )
        .agg(
            low_traffic_percent=(
                "low",
                "mean",
            ),
            points=(
                "low",
                "size",
            ),
        )
        .sort_values(
            [
                "low_traffic_percent",
                "points",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )


# ----------------------------
# LEFT / BASELINE MAP
# ----------------------------
def build_left_map(
    baseline_train,
    max_gap_min,
):

    if baseline_train.empty:

        return folium.Map(
            location=[
                27.9,
                -82.4,
            ],
            zoom_start=7,
            tiles="OpenStreetMap",
        )

    m = folium.Map(
        location=[
            float(
                baseline_train[
                    "LAT"
                ].median()
            ),
            float(
                baseline_train[
                    "LON"
                ].median()
            ),
        ],
        zoom_start=7,
        tiles="OpenStreetMap",
    )

    total = 0

    base_color = (
        "#2b6cb0"
    )

    vessel_order = (
        baseline_train
        .groupby(
            "MMSI"
        )
        .size()
        .sort_values(
            ascending=False
        )
        .index
        .tolist()[
            :BASELINE_MAX_VESSELS
        ]
    )

    for mmsi in vessel_order:

        g = (
            baseline_train[
                baseline_train[
                    "MMSI"
                ]
                == mmsi
            ]
            .sort_values(
                "BaseDateTime"
            )
        )

        if g.empty:
            continue

        g = downsample(
            g,
            BASELINE_TARGET_POINTS_PER_MMSI,
        )

        chunk = []
        last_t = None

        for row in g.itertuples(
            index=False
        ):

            t = (
                row.BaseDateTime
            )

            point = (
                float(
                    row.LAT
                ),
                float(
                    row.LON
                ),
            )

            if (
                last_t is not None
                and
                (
                    t - last_t
                ).total_seconds()
                / 60
                > max_gap_min
            ):

                if len(chunk) >= 2:

                    folium.PolyLine(
                        chunk,
                        color=base_color,
                        weight=3,
                        opacity=0.35,
                    ).add_to(m)

                    total += len(
                        chunk
                    )

                chunk = []

            chunk.append(
                point
            )

            last_t = t

        if len(chunk) >= 2:

            folium.PolyLine(
                chunk,
                color=base_color,
                weight=3,
                opacity=0.35,
            ).add_to(m)

            total += len(
                chunk
            )

        # Always show at least one point
        # for each vessel.
        last = g.iloc[-1]

        folium.CircleMarker(
            location=[
                float(
                    last[
                        "LAT"
                    ]
                ),
                float(
                    last[
                        "LON"
                    ]
                ),
            ],
            radius=2,
            color=base_color,
            fill=True,
            fill_opacity=0.55,
            tooltip=f"MMSI: {mmsi}",
        ).add_to(m)

        if (
            total
            >= BASELINE_MAX_TOTAL_POINTS
        ):
            break

    fit_map_to_data(
        m,
        baseline_train,
    )

    return m


@st.cache_data(
    show_spinner=False
)
def left_map_html_cached(
    baseline_train,
    max_gap_min,
):

    return (
        build_left_map(
            baseline_train,
            max_gap_min,
        )
        .get_root()
        .render()
    )


# ----------------------------
# RIGHT / TRACKS MAP
# ----------------------------
def build_right_map(
    tracks,
    max_gap_min,
):

    if tracks.empty:

        return folium.Map(
            location=[
                27.9,
                -82.4,
            ],
            zoom_start=7,
            tiles="OpenStreetMap",
        )

    m = folium.Map(
        location=[
            float(
                tracks[
                    "LAT"
                ].median()
            ),
            float(
                tracks[
                    "LON"
                ].median()
            ),
        ],
        zoom_start=7,
        tiles="OpenStreetMap",
    )

    total = 0

    vessel_order = (
        tracks
        .groupby(
            "MMSI"
        )
        .size()
        .sort_values(
            ascending=False
        )
        .index
        .tolist()[
            :RIGHT_MAX_VESSELS
        ]
    )

    for mmsi in vessel_order:

        g = (
            tracks[
                tracks[
                    "MMSI"
                ]
                == mmsi
            ]
            .sort_values(
                "BaseDateTime"
            )
        )

        if g.empty:
            continue

        g = downsample(
            g,
            RIGHT_POINTS_PER_VESSEL,
        )

        color = color_for(
            mmsi
        )

        chunk = []
        last_t = None

        for row in g.itertuples(
            index=False
        ):

            t = (
                row.BaseDateTime
            )

            point = (
                float(
                    row.LAT
                ),
                float(
                    row.LON
                ),
            )

            if (
                last_t is not None
                and
                (
                    t - last_t
                ).total_seconds()
                / 60
                > max_gap_min
            ):

                if len(chunk) >= 2:

                    folium.PolyLine(
                        chunk,
                        color=color,
                        weight=4,
                        opacity=0.90,
                        tooltip=f"MMSI: {mmsi}",
                        popup=f"MMSI: {mmsi}",
                    ).add_to(m)

                    total += len(
                        chunk
                    )

                chunk = []

            chunk.append(
                point
            )

            last_t = t

        if len(chunk) >= 2:

            folium.PolyLine(
                chunk,
                color=color,
                weight=4,
                opacity=0.90,
                tooltip=f"MMSI: {mmsi}",
                popup=f"MMSI: {mmsi}",
            ).add_to(m)

            total += len(
                chunk
            )

        # Always show the newest vessel point.
        last = g.iloc[-1]

        folium.CircleMarker(
            location=[
                float(
                    last[
                        "LAT"
                    ]
                ),
                float(
                    last[
                        "LON"
                    ]
                ),
            ],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.9,
            tooltip=f"MMSI: {mmsi}",
            popup=f"MMSI: {mmsi}",
        ).add_to(m)

        if (
            total
            >= RIGHT_MAX_TOTAL_POINTS
        ):
            break

    fit_map_to_data(
        m,
        tracks,
    )

    return m


# ----------------------------
# SIDEBAR CONTROLS
# ----------------------------
st.sidebar.markdown(
    "## AIS Filters"
)

st.sidebar.caption(
    "AIS data source: GitHub"
)

st.session_state[
    "ais_class_mode"
] = st.sidebar.selectbox(
    "AIS Class (RIGHT map only)",
    [
        "All",
        "Class A",
        "Class B",
    ],
    index=[
        "All",
        "Class A",
        "Class B",
    ].index(
        st.session_state[
            "ais_class_mode"
        ]
    ),
)

sog_min = st.sidebar.number_input(
    "SOG min",
    min_value=0.0,
    max_value=20.0,
    value=0.0,
    step=0.5,
)

max_gap_min = st.sidebar.number_input(
    "Max time gap (minutes)",
    min_value=1,
    max_value=1440,
    value=60,
    step=5,
)

train_frac = st.sidebar.number_input(
    "Train fraction",
    min_value=0.50,
    max_value=0.95,
    value=0.80,
    step=0.05,
)

grid_size = st.sidebar.number_input(
    "Grid size (scoring)",
    min_value=0.0005,
    max_value=0.02,
    value=0.0012,
    step=0.0001,
    format="%.4f",
)

low_pct = st.sidebar.number_input(
    "Low-traffic percentile",
    min_value=0.01,
    max_value=0.30,
    value=0.08,
    step=0.01,
)

min_pts = st.sidebar.number_input(
    "Min pings per vessel",
    min_value=2,
    max_value=10000,
    value=10,
    step=1,
)

top_k = st.sidebar.number_input(
    "Max vessels",
    min_value=1,
    max_value=100,
    value=5,
    step=1,
)

show_only_topk = (
    st.sidebar.checkbox(
        "Right map: show ONLY top choices",
        value=True,
    )
)


# ----------------------------
# LOAD DATA
# ----------------------------
try:

    ais = load_ais(
        AIS_DATA_URL
    )

except Exception as exc:

    st.error(
        f"AIS data could not be loaded: {exc}"
    )

    st.stop()


if ais.empty:

    st.error(
        "No AIS rows were loaded. "
        "Check Florida_routes.csv and its column names."
    )

    st.stop()


# ----------------------------
# PIPELINE
# ----------------------------

# Important:
# We DO NOT delete AIS rows based on time gaps.
# Gaps are handled only while drawing routes.

baseline_tracks = ais[
    ais["SOG"]
    >= BASELINE_SOG_MIN
].copy()


baseline_train, _ = split_by_time(
    baseline_tracks,
    BASELINE_TRAIN_FRACTION,
)


user_df = filter_class(
    ais,
    st.session_state[
        "ais_class_mode"
    ],
)


user_df = user_df[
    user_df["SOG"]
    >= float(
        sog_min
    )
].copy()


_, user_test = split_by_time(
    user_df,
    float(
        train_frac
    ),
)


baseline_grid = make_baseline_grid(
    baseline_train,
    float(
        grid_size
    ),
)


scores_all = score_everyone(
    user_test,
    baseline_grid,
    float(
        grid_size
    ),
    float(
        low_pct
    ),
)


scores = scores_all[
    scores_all[
        "points"
    ]
    >= int(
        min_pts
    )
].copy()


suspect_mmsi = (
    scores
    .head(
        int(
            top_k
        )
    )[
        "MMSI"
    ]
    .tolist()
    if not scores.empty
    else []
)


# If no vessel passes the outlier scoring filters,
# show the vessels with the most pings instead.
# This prevents the right map from being empty.

used_fallback = False


if show_only_topk:

    if suspect_mmsi:

        tracks_show = user_test[
            user_test[
                "MMSI"
            ].isin(
                suspect_mmsi
            )
        ].copy()

    else:

        used_fallback = True

        fallback_mmsi = (
            user_test
            .groupby(
                "MMSI"
            )
            .size()
            .sort_values(
                ascending=False
            )
            .head(
                int(
                    top_k
                )
            )
            .index
            .tolist()
            if not user_test.empty
            else []
        )

        tracks_show = user_test[
            user_test[
                "MMSI"
            ].isin(
                fallback_mmsi
            )
        ].copy()

else:

    tracks_show = (
        user_test.copy()
    )


# ----------------------------
# DATA CHECK
# ----------------------------
st.sidebar.markdown(
    "---"
)

st.sidebar.markdown(
    "### Data Check"
)

st.sidebar.write(
    f"AIS rows loaded: {len(ais):,}"
)

st.sidebar.write(
    f"Unique vessels: {ais['MMSI'].nunique():,}"
)

st.sidebar.write(
    f"Baseline rows: {len(baseline_train):,}"
)

st.sidebar.write(
    f"Filtered test rows: {len(user_test):,}"
)

st.sidebar.write(
    f"Qualified vessels: {len(scores):,}"
)

st.sidebar.write(
    f"Rows sent to right map: {len(tracks_show):,}"
)


if used_fallback:

    st.sidebar.warning(
        "No vessels met the scoring filters. "
        "The right map is showing the vessels "
        "with the most AIS pings instead."
    )


# ----------------------------
# MAPS
# ----------------------------
col1, col2 = st.columns(
    2
)


with col1:

    st.subheader(
        "Baseline"
    )

    left_html = left_map_html_cached(
        baseline_train,
        int(
            max_gap_min
        ),
    )

    components.html(
        left_html,
        height=650,
        scrolling=True,
    )


with col2:

    st.subheader(
        "Tracks of Interest"
    )

    mmsi_list = (
        sorted(
            tracks_show[
                "MMSI"
            ]
            .unique()
            .tolist()
        )
        if not tracks_show.empty
        else []
    )

    chosen = st.selectbox(
        "Focus MMSI (RIGHT map)",
        [
            "All (Top)"
        ]
        + [
            str(x)
            for x
            in mmsi_list
        ],
        index=0,
    )

    tracks_right = (
        tracks_show.copy()
    )

    if (
        chosen
        != "All (Top)"
    ):

        tracks_right = tracks_show[
            tracks_show[
                "MMSI"
            ]
            == int(
                chosen
            )
        ].copy()

    state = show_map(
        build_right_map(
            tracks_right,
            int(
                max_gap_min
            ),
        )
    )

    if isinstance(
        state,
        dict,
    ):

        clicked_text = (
            str(
                state.get(
                    "last_object_clicked_tooltip",
                    "",
                )
            )
            + " "
            + str(
                state.get(
                    "last_object_clicked_popup",
                    "",
                )
            )
        )

        match = re.search(
            r"MMSI\s*:\s*(\d+)",
            clicked_text,
        )

        if match:

            st.session_state[
                "clicked_mmsi"
            ] = (
                match.group(1)
            )


# ----------------------------
# SELECTED MMSI + IDENTIFY
# ----------------------------
st.markdown(
    "---"
)

st.subheader(
    "Selected MMSI"
)


if st.session_state[
    "clicked_mmsi"
]:

    mmsi = st.session_state[
        "clicked_mmsi"
    ]

    st.write(
        f"**Clicked MMSI:** `{mmsi}`"
    )

else:

    st.caption(
        "Click a colored track or marker "
        "on the RIGHT map to select its MMSI."
    )


st.markdown(
    """
    <a
        href="https://www.marinetraffic.com/en/ais/details/ships"
        target="_blank"
        style="text-decoration:none;"
    >
      <div style="
        margin-top:10px;
        padding:18px;
        border-radius:12px;
        background:#0e3a5b;
        color:white;
        font-size:20px;
        font-weight:700;
        text-align:center;
        cursor:pointer;
      ">
        Identify
      </div>
    </a>
    """,
    unsafe_allow_html=True,
)


# ----------------------------
# DOWNLOAD CURRENT RIGHT-MAP DATA
# ----------------------------
EXPORT_COLS = [
    "MMSI",
    "VesselName",
    "BaseDateTime",
    "LAT",
    "LON",
    "SOG",
    "COG",
    "Status",
    "TransceiverClass",
]


st.markdown(
    "<div style='height:10px;'></div>",
    unsafe_allow_html=True,
)


if (
    isinstance(
        tracks_right,
        pd.DataFrame,
    )
    and not tracks_right.empty
):

    export_df = (
        tracks_right.copy()
    )

    for col in EXPORT_COLS:

        if col not in export_df.columns:
            export_df[col] = ""

    export_df = export_df[
        EXPORT_COLS
    ].copy()

    export_df[
        "BaseDateTime"
    ] = (
        pd.to_datetime(
            export_df[
                "BaseDateTime"
            ],
            errors="coerce",
        )
        .dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    csv_bytes = (
        export_df
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )

    st.download_button(
        label="Download displayed AIS tracks",
        data=csv_bytes,
        file_name="olaf_tracks_of_interest.csv",
        mime="text/csv",
        use_container_width=True,
    )

else:

    st.info(
        "No right-map AIS tracks are currently "
        "available to download."
    )
