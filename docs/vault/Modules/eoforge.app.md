---
tags: [module, zoo]
---
# eoforge.app

The Streamlit web app (`eoforge/app/app.py`), started with `eoforge app --folder <folder> [--port 8501]`. See [[Model zoo and app]].

- Lists every GeoTIFF under the folder (outputs excluded); uploads go to `<folder>/uploads` (a different file with an existing name gets a new name).
- Band matching as in [[run_model]]: by name, stand-ins, [[guess_descriptions]] for unnamed bands; any band can be set by hand.
- Run options: device, probabilities, tile size, *Subtract offset* (Sentinel-2 since 2022) and *Resample to the model's pixel size*.
- Caches the loaded model between runs (up to 3), so a second run of the same model starts immediately.
- Reads results back from `eoforge_results/*/summary.json`, so earlier runs can be reopened; warnings from the run are shown above the legend.
- Map: folium with Esri World Imagery and OpenStreetMap basemaps; the input image, result and uncertainty are embedded as PNG overlays.
- *Add a model* tab writes ONNX cards with `write_card`.

Testing tip: run it with `streamlit.testing.v1.AppTest.from_file(path)`; importing the module directly executes the page outside Streamlit and leaves a form open, which breaks later widgets.
