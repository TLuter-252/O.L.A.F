# O.L.A.F. Streamlit v2

This folder is a Streamlit Community Cloud-ready AIS track outlier application.

## Deploy

In Streamlit Community Cloud, select this repository and set the main file path to:

`olaf_streamlit_v2/app.py`

The hosted demo uses `../Florida_routes.csv` and `../Olaf.png` from the repository. Analysts can also upload an AIS CSV or ZIP at runtime.

## What it does

- Left map: every drawable vessel track in the loaded dataset (no point markers), rendered as dense blue lines.
- Right map: exactly the top one or two vessel tracks under the analyst's settings.
- Synchronized navigation: panning or zooming either map moves the other to the same view.
- Initial view: automatically centers on the region crossed by the most unique vessels.
- Basemap: standard OpenStreetMap tiles, with no API key required.
- Ranking: combines route rarity, unusual speed, and unusual course with adjustable weights.
- Track continuity: breaks lines at analyst-defined time gaps so vessels are not joined across missing observations.

Run locally from the repository root:

```powershell
pip install -r olaf_streamlit_v2/requirements.txt
streamlit run olaf_streamlit_v2/app.py
```
