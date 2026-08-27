# O.L.A.F. Streamlit v2

This folder is a Streamlit Community Cloud-ready AIS track outlier application.

## Deploy

In Streamlit Community Cloud, select this repository and set the main file path to:

`olaf_streamlit_v2/app.py`

The hosted demo uses a compact, full-day July 4, 2023 AIS track dataset in
`data/ais_2023_07_04_tracks.csv.gz` and `../Olaf.png` from the repository. July 4
was selected after comparing the available days because it had the most vessels,
the most long briefing-quality tracks, and the most useful tracks in its busiest
region. Analysts can also upload an AIS CSV or ZIP at runtime.

## What it does

- Left map: every complete vessel track crossing the dataset's busiest region (no point
  markers), which maximizes visible track density without sending thousands of off-screen
  national tracks to the browser.
- Right map: exactly the top one or two long vessel tracks under the analyst's settings.
- Synchronized navigation: panning or zooming either map moves the other to the same view.
- Initial view: automatically centers on the region crossed by the most unique vessels.
- Basemap: standard OpenStreetMap tiles, with no API key required.
- Ranking: combines route rarity, unusual speed, and unusual course with adjustable weights,
  then favors complete tracks spanning at least 6 hours and 30 km.
- Track continuity: breaks lines at analyst-defined time gaps so vessels are not joined across missing observations.

Run locally from the repository root:

```powershell
pip install -r olaf_streamlit_v2/requirements.txt
streamlit run olaf_streamlit_v2/app.py
```
