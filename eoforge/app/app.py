"""EOForge web app: run ready-made models on your imagery and look at the result on a map.

Start with:  eoforge app --folder <folder with your images>     (then open http://localhost:8501)
"""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import numpy as np
import rasterio
import streamlit as st

from eoforge.zoo import BandError, load_adapter, run_model, trained_cards, write_card
from eoforge.zoo.adapters import ADAPTERS, missing_packages
from eoforge.zoo.bands import canonical, guess_descriptions, match_bands, prefixes
from eoforge.zoo.card import load_cards

ROOT = Path(os.environ.get("EOFORGE_FOLDER", os.getcwd())).resolve()
RESULTS = ROOT / "eoforge_results"
SKIP_NAMES = {"classes.tif", "entropy.tif", "probabilities.tif", "objects.tif"}

st.set_page_config(page_title="EOForge", page_icon="🛰️", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea { font-family: 'IBM Plex Sans', sans-serif; }
h1 { font-weight: 600; letter-spacing: -0.01em; margin-bottom: 0.1rem; }
h2, h3 { font-weight: 600; }
.block-container { padding-top: 1.6rem; max-width: 1500px; }
.legend-row { display: flex; align-items: center; gap: 0.55rem; padding: 0.22rem 0; font-size: 0.95rem; }
.legend-swatch { width: 1.05rem; height: 1.05rem; border-radius: 3px; border: 1px solid rgba(31,42,44,0.25); flex: none; }
.legend-name { flex: 1; }
.legend-num { font-variant-numeric: tabular-nums; color: #4a5759; min-width: 4.5rem; text-align: right; }
.muted { color: #5d6b6d; font-size: 0.92rem; }
.card-meta { font-size: 0.9rem; color: #3d4a4c; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- data helpers
@st.cache_data(ttl=30)
def find_images(root: str) -> list[str]:
    out = []
    for p in Path(root).rglob("*.tif*"):
        rel = p.relative_to(root)
        if p.name in SKIP_NAMES or "eoforge_results" in rel.parts or p.suffix.lower() not in (".tif", ".tiff"):
            continue
        if any(part.startswith(".") for part in rel.parts):
            continue
        out.append(str(rel))
        if len(out) >= 500:
            break
    return sorted(out)


@st.cache_data(ttl=60)
def image_info(path: str):
    with rasterio.open(path) as s:
        return {"count": s.count, "desc": [d or "" for d in s.descriptions], "width": s.width, "height": s.height,
                "crs": str(s.crs) if s.crs else "none", "res": abs(s.transform.a),
                "projected": bool(s.crs and s.crs.is_projected), "dtype": s.dtypes[0]}


@st.cache_data(ttl=30)
def all_cards(root: str):
    cards = load_cards()
    cards.update(trained_cards([Path(root)]))
    return cards, list(load_cards.errors)


@st.cache_resource(max_entries=3, show_spinner=False)
def cached_adapter(card_id: str, card_path: str, device: str, _card):
    return load_adapter(_card, device, log=lambda m: None)


def past_runs():
    if not RESULTS.is_dir():
        return []
    runs = [p for p in RESULTS.iterdir() if (p / "summary.json").exists()]
    return sorted(runs, key=lambda p: (p / "summary.json").stat().st_mtime, reverse=True)


# --------------------------------------------------------------------------- map
def data_uri(path: Path) -> str:
    import base64
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def show_map(run: Path, opacity: float, show_entropy: bool):
    import folium
    from streamlit_folium import st_folium
    meta = json.loads((run / "preview.json").read_text())
    (s, w), (n, e) = meta["result"]["bounds"]
    m = folium.Map(location=[(s + n) / 2, (w + e) / 2], zoom_start=13, tiles=None, control_scale=True)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri, Maxar, Earthstar Geographics", name="Satellite basemap").add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street map", show=False).add_to(m)
    folium.raster_layers.ImageOverlay(data_uri(run / meta["image"]["file"]), bounds=meta["image"]["bounds"],
                                      name="Input image", opacity=1.0, interactive=False).add_to(m)
    folium.raster_layers.ImageOverlay(data_uri(run / meta["result"]["file"]), bounds=meta["result"]["bounds"],
                                      name="Model result", opacity=opacity, interactive=False).add_to(m)
    if "entropy" in meta:
        folium.raster_layers.ImageOverlay(data_uri(run / meta["entropy"]["file"]), bounds=meta["entropy"]["bounds"],
                                          name="Uncertainty", opacity=0.85, show=show_entropy).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[s, w], [n, e]])
    st_folium(m, height=640, use_container_width=True, returned_objects=[], key=f"map-{run.name}-{opacity}-{show_entropy}")


def legend(summary: dict, cards: dict):
    card = cards.get(summary["model"]["id"])
    colours = {c.value: c.colour for c in card.classes} if card else {}
    areas = summary["areas"]
    if isinstance(areas, dict):          # objects
        st.markdown(f"**{areas['objects']:,} objects** outlined, covering {100 * areas['covered_fraction']:.1f}% of the image.")
        return
    has_ha = any("hectares" in a for a in areas)
    rows = []
    for a in areas:
        sw = colours.get(a["value"], "#999999")
        ha = f"{a['hectares']:,.1f} ha" if has_ha else ""
        rows.append(f"<div class='legend-row'><span class='legend-swatch' style='background:{sw}'></span>"
                    f"<span class='legend-name'>{a['name']}</span><span class='legend-num'>{a['percent']:.1f}%</span>"
                    f"<span class='legend-num'>{ha}</span></div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


def result_panel(run: Path, cards: dict):
    summary = json.loads((run / "summary.json").read_text())
    left, right = st.columns([3, 1.15], gap="large")
    with right:
        st.subheader(summary["model"]["name"])
        st.markdown(f"<div class='muted'>{Path(summary['image']).name}, {summary['runtime_s']} s on {summary['device']}</div>",
                    unsafe_allow_html=True)
        for w in summary.get("warnings", []):
            st.warning(w)
        info_ = summary.get("info", {})
        if info_.get("note"):
            st.info(info_["note"])
        if "fields" in info_:
            st.markdown(f"**{info_['fields']:,} field polygons** saved as {info_.get('vector', 'a GeoPackage')}.")
        opacity = st.slider("Result opacity", 0.0, 1.0, 0.75, 0.05, key=f"op-{run.name}")
        show_entropy = st.toggle("Show uncertainty layer", value=False, key=f"ent-{run.name}",
                                 disabled=not (run / "entropy.tif").exists())
        legend(summary, cards)
        st.markdown("")
        for f, label in [("classes.tif", "Download class map"), ("objects.tif", "Download objects raster"),
                         ("objects.gpkg", "Download objects (GeoPackage)"), ("fields.gpkg", "Download field polygons (GeoPackage)"), ("entropy.tif", "Download uncertainty"),
                         ("summary.json", "Download summary")]:
            if (run / f).exists():
                st.download_button(label, (run / f).read_bytes(), file_name=f"{run.name}_{f}", key=f"dl-{run.name}-{f}",
                                   use_container_width=True)
        st.caption(f"Saved in {run}")
    with left:
        if (run / "preview.json").exists():
            show_map(run, opacity, show_entropy)
        else:
            st.info("No map preview for this run; the GeoTIFFs are in the results folder.")


# --------------------------------------------------------------------------- page
st.title("EOForge")
st.markdown("<div class='muted'>Run ready-made Earth observation models on your own imagery.</div>", unsafe_allow_html=True)
tab_run, tab_models, tab_add = st.tabs(["Run", "Models", "Add a model"])
cards, card_errors = all_cards(str(ROOT))

with st.sidebar:
    st.header("1. Image")
    images = find_images(str(ROOT))
    up = st.file_uploader("Add a GeoTIFF", type=["tif", "tiff"])
    if up is not None and st.session_state.get("uploaded_id") != up.file_id:
        # handle each upload once, so the image list can still be used while the file stays in the uploader
        dst = ROOT / "uploads" / up.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size != up.size:          # a different file with the same name
            stem, n = dst.stem, 2
            while dst.exists():
                dst = dst.with_name(f"{stem}_{n}{dst.suffix}")
                n += 1
        if not dst.exists():
            dst.write_bytes(up.getbuffer())
        find_images.clear()
        images = find_images(str(ROOT))
        st.session_state["uploaded_id"] = up.file_id
        st.session_state["image"] = str(dst.relative_to(ROOT))
    if not images:
        st.info(f"No GeoTIFFs found in {ROOT}. Add one above, or start the app with --folder pointing at your images.")
    image_rel = st.selectbox("Image", images, index=images.index(st.session_state["image"])
                             if st.session_state.get("image") in images else 0, key="image") if images else None
    info = image_info(str(ROOT / image_rel)) if image_rel else None
    if info:
        st.markdown(f"<div class='card-meta'>{info['width']:,} × {info['height']:,} px, {info['count']} bands, "
                    f"{info['res']:g} {'m' if info['projected'] else '°'} pixels, {info['crs']}</div>", unsafe_allow_html=True)

    st.header("2. Model")
    order = ["dynamic-world", "terramind-lulc", "ftw-fields", "ftw-fields-fast", "ftw-fields-1date", "delineate-anything",
             "delineate-anything-small", "prithvi-floods", "prithvi-burnscars", "omnicloudmask", "sam2"]
    ids = sorted(cards, key=lambda i: (order.index(i) if i in order else len(order), i))
    labels = {i: cards[i].name for i in ids}
    model_id = st.selectbox("Model", ids, format_func=lambda i: labels[i], key="model") if ids else None
    card = cards.get(model_id)
    ready = False
    if card:
        st.markdown(f"<div class='card-meta'>{card.description}</div>", unsafe_allow_html=True)
        if card.sensor:
            st.markdown(f"<div class='card-meta'><b>Expects</b> {card.sensor}"
                        + (f", {card.resolution_m:g} m pixels" if card.resolution_m else "") + "</div>",
                        unsafe_allow_html=True)
        miss = missing_packages(ADAPTERS[card.adapter]) if card.adapter in ADAPTERS else ["unknown adapter"]
        if miss:
            st.warning(f"Needs {', '.join(miss)}. In your terminal run:\n\n`{card.install or 'pip install ' + ' '.join(miss)}`")

    st.header("3. Bands")
    bands_sel, windows_sel = None, None
    fallback = dict(card.params.get("band_fallback", {})) if card else {}

    def auto_bands(desc, prefix):
        """Match the card's bands by name, using the card's fallbacks for missing ones."""
        desc, guess = guess_descriptions(desc, card.bands)
        auto, missing = match_bands(card.bands, desc, prefix)
        notes = [guess] if guess else []
        for i, need in enumerate(card.bands):
            if auto[i] is None and need in fallback:
                fb, _ = match_bands([fallback[need]], desc, prefix)
                if fb[0] is not None:
                    auto[i] = fb[0]
                    notes.append(f"{need} is not in this image; {fallback[need]} is used instead.")
        return auto, [b for b, a in zip(card.bands, auto) if a is None], notes

    if card and info:
        n_windows = int(card.params.get("windows", 1))
        if card.bands and n_windows > 1:
            st.markdown("<div class='muted'>This model needs the same bands at two dates: the start and the end "
                        "of the growing season. Use two band sets of one image (wet_ and dry_ in seasonal stacks) "
                        "or two images.</div>", unsafe_allow_html=True)
            windows_sel = []
            for wi, label in enumerate(["Start of season", "End of season"][:n_windows]):
                st.markdown(f"**{label}**")
                w_img = st.selectbox("Image", images, index=images.index(image_rel), key=f"win-img-{wi}-{model_id}")
                w_info = image_info(str(ROOT / w_img))
                w_pre = prefixes(w_info["desc"])
                default = wi if wi < len(w_pre) and any(w_pre) else 0
                w_prefix = st.selectbox("Band set", w_pre, index=default, format_func=lambda p: p or "unprefixed",
                                        key=f"win-pre-{wi}-{model_id}-{w_img}")
                auto, missing, notes = auto_bands(w_info["desc"], w_prefix)
                if missing:
                    st.warning(f"No band named {', '.join(missing)} in this band set.")
                    windows_sel = None
                elif windows_sel is not None:
                    names = [w_info["desc"][b - 1] or f"band {b}" for b in auto]
                    st.markdown(f"<div class='card-meta'>{', '.join(names)}</div>", unsafe_allow_html=True)
                    windows_sel.append({"image": str(ROOT / w_img), "prefix": w_prefix, "bands": auto})
            if windows_sel and len({(w["image"], w["prefix"]) for w in windows_sel}) < len(windows_sel):
                st.warning("Both dates use the same bands. The model works best with two different dates.")
        elif card.bands:
            pre = prefixes(info["desc"])
            prefix = None
            if len(pre) > 1 or (pre and pre[0]):
                prefix = st.selectbox("Use bands from", pre, format_func=lambda p: p or "unprefixed", key="prefix")
            auto, missing, notes = auto_bands(info["desc"], prefix)
            options = [None] + list(range(1, info["count"] + 1))
            fmt = lambda b: "not set" if b is None else f"{b}: {info['desc'][b - 1] or 'no name'}"
            bands_sel = []
            cols = st.columns(2)
            for i, need in enumerate(card.bands):
                with cols[i % 2]:
                    bands_sel.append(st.selectbox(need, options, index=options.index(auto[i]), format_func=fmt,
                                                  key=f"band-{model_id}-{image_rel}-{need}-{prefix}"))
            if notes:
                st.info("  \n".join(notes))
            if missing:
                st.warning(f"This image has no band named {', '.join(missing)}. Choose the matching band for each, "
                           "or pick an image with these bands.")
        else:
            st.markdown("<div class='muted'>This model uses all bands.</div>", unsafe_allow_html=True)
        if card.resolution_m and info["projected"] and abs(info["res"] - card.resolution_m) / card.resolution_m > 0.25:
            st.warning(f"This image has {info['res']:g} m pixels; the model was made for {card.resolution_m:g} m.")
        ready = not (card.adapter in ADAPTERS and missing_packages(ADAPTERS[card.adapter]))
        if n_windows > 1 and card.bands:
            ready = ready and bool(windows_sel)
        else:
            ready = ready and (bands_sel is None or all(b is not None for b in bands_sel))

    st.header("4. Run")
    with st.expander("Options"):
        device = st.selectbox("Device", ["auto", "cuda", "cpu"], help="auto uses the GPU when PyTorch sees one")
        write_probs = st.checkbox("Also save class probabilities", value=False)
        tile = st.number_input("Tile size (pixels, 0 = model default)", min_value=0, max_value=4096, value=0, step=64)
        offset = st.number_input("Subtract offset", min_value=0, max_value=5000, value=0, step=1000,
                                 help="1000 for Sentinel-2 L2A/L1C processed with baseline 04.00+ (data from 25 Jan 2022) "
                                      "when the provider has not removed the offset (e.g. Planetary Computer). "
                                      "Earth Engine 'HARMONIZED' collections already removed it.")
        match_res = st.checkbox("Resample to the model's pixel size", value=False,
                                help="e.g. 10 m Sentinel-2 to 30 m for the HLS-trained burn-scar model")
    go = st.button("Run model", type="primary", use_container_width=True, disabled=not (ready and info))

with tab_run:
    if go and card and info:
        prog = st.progress(0.0, text=f"Loading {card.name} ...")
        log_box = st.empty()
        lines = []

        def log(msg):
            lines.append(msg)
            log_box.caption(msg)

        try:
            adapter = cached_adapter(card.id, str(card.path), device, card)
            prog.progress(0.05, text=f"Running {card.name} on {Path(image_rel).name} ...")
            opts = {"write_probabilities": write_probs}
            if offset:
                opts["offset"] = float(offset)
            if match_res:
                opts["match_resolution"] = True
            if tile:
                opts["tile"] = int(tile)
            summary = run_model(card, ROOT / image_rel, RESULTS, bands=bands_sel, windows=windows_sel, device=device, options=opts,
                                adapter=adapter, log=log, progress=lambda f: prog.progress(min(0.05 + 0.9 * f, 0.95),
                                                                                         text=f"Running {card.name} ... {f:.0%}"))
            prog.progress(1.0, text="Done")
            st.session_state["show_run"] = Path(summary["folder"]).name
            log_box.empty()
        except BandError as e:
            prog.empty()
            st.error(f"The image is missing bands the model needs: {', '.join(e.missing)}. Pick them in step 3.")
        except ImportError as e:
            prog.empty()
            st.error(str(e))
        except Exception as e:
            prog.empty()
            st.error(f"The run stopped: {e}")
            with st.expander("Details"):
                st.code(traceback.format_exc())

    runs = past_runs()
    if runs:
        names = [r.name for r in runs]
        current = st.session_state.get("show_run", names[0])
        choice = st.selectbox("Result", names, index=names.index(current) if current in names else 0,
                              format_func=lambda n: n.replace("__", "  with  "), key="result_pick")
        result_panel(RESULTS / choice, cards)
    else:
        st.markdown("### Pick an image and a model, then run it")
        st.markdown("<div class='muted'>Results appear here on a map and are saved, with their GeoTIFFs, in "
                    f"<code>{RESULTS}</code>.</div>", unsafe_allow_html=True)

with tab_models:
    st.markdown("<div class='muted'>Models come from model cards: the built-in ones, your own cards in "
                "<code>~/.eoforge/models</code>, and every <code>best.pt</code> you trained under this folder.</div>",
                unsafe_allow_html=True)
    for e in card_errors:
        st.error(f"A model card could not be read: {e}")
    for c in cards.values():
        with st.expander(f"{c.name}"):
            a, b = st.columns([2, 1])
            with a:
                st.markdown(c.description or "")
                if c.notes:
                    st.markdown(f"<div class='muted'>{c.notes}</div>", unsafe_allow_html=True)
                if c.classes:
                    st.markdown("".join(f"<div class='legend-row'><span class='legend-swatch' style='background:{x.colour}'></span>"
                                        f"<span class='legend-name'>{x.name}</span></div>" for x in c.classes), unsafe_allow_html=True)
            with b:
                miss = missing_packages(ADAPTERS[c.adapter]) if c.adapter in ADAPTERS else ["adapter missing"]
                st.markdown(f"<div class='card-meta'><b>Bands</b> {', '.join(c.bands) or 'all'}<br>"
                            f"<b>Sensor</b> {c.sensor or '–'}<br><b>Pixel size</b> {c.resolution_m or '–'} m<br>"
                            f"<b>Runs with</b> {c.adapter}<br><b>Licence</b> {c.licence or '–'}<br>"
                            f"<b>Status</b> {'ready' if not miss else 'needs ' + ', '.join(miss)}</div>", unsafe_allow_html=True)
                if c.source:
                    st.markdown(f"[Source]({c.source})" if c.source.startswith("http") else f"<div class='muted'>{c.source}</div>",
                                unsafe_allow_html=True)
                if c.citation:
                    st.caption(c.citation)

with tab_add:
    st.markdown("Add a model you found as an **ONNX file**. For models that need their own code (TerraTorch configs, "
                "Hugging Face pipelines), copy one of the built-in cards in `eoforge/zoo/cards` and edit it; "
                "the guide in the docs vault explains each field.")
    with st.form("add_model"):
        c1, c2 = st.columns(2)
        with c1:
            new_id = st.text_input("Short id", placeholder="kaza-landcover-v1")
            new_name = st.text_input("Name", placeholder="KAZA land cover U-Net")
            weights = st.text_input("ONNX file (path or https URL)", placeholder="/path/to/model.onnx")
            bands_txt = st.text_input("Bands, in the model's order", placeholder="B02, B03, B04, B08, B11, B12")
            res_m = st.number_input("Pixel size the model expects (m, 0 = any)", 0.0, 1000.0, 10.0)
        with c2:
            layout = st.selectbox("Input layout", ["NCHW", "NHWC"], help="NCHW for PyTorch exports, NHWC for most TensorFlow ones")
            norm = st.selectbox("Normalisation", ["scale", "standard", "minmax", "dynamic_world", "none"])
            norm_vals = st.text_input("Normalisation values",
                                      placeholder="scale: 0.0001   |   standard: mean1,mean2,... ; std1,std2,...")
            output = st.selectbox("Model output", ["logits", "probs"])
            classes_txt = st.text_area("Classes, one per line: value, name, colour",
                                       placeholder="0, water, #419bdf\n1, woodland, #397d49\n2, grassland, #88b053")
        submitted = st.form_submit_button("Save model card")
    if submitted:
        try:
            bands = [canonical(b) for b in bands_txt.replace(";", ",").split(",") if b.strip()]
            classes = []
            for line in classes_txt.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    classes.append({"value": int(parts[0]), "name": parts[1], "colour": parts[2] if len(parts) > 2 else "#888888"})
            spec = {"type": norm}
            if norm == "scale":
                spec["factor"] = float(norm_vals or 1)
            elif norm in ("standard", "minmax"):
                a_, b_ = norm_vals.split(";")
                keys = ("mean", "std") if norm == "standard" else ("min", "max")
                spec[keys[0]] = [float(v) for v in a_.split(",")]
                spec[keys[1]] = [float(v) for v in b_.split(",")]
                if len(spec[keys[0]]) != len(bands) or len(spec[keys[1]]) != len(bands):
                    raise ValueError(f"Give one {keys[0]} and one {keys[1]} value per band ({len(bands)} bands)")
            w = {"url": weights} if weights.startswith("http") else {"path": str(Path(weights).expanduser().resolve())}
            card_dict = {"id": new_id.strip(), "name": new_name.strip() or new_id.strip(), "adapter": "onnx",
                         "task": "segmentation", "bands": bands, "classes": classes,
                         "params": {"layout": layout, "normalise": spec, "output": output, "tile": 512, "overlap": 32},
                         "weights": w, "source": weights}
            if res_m:
                card_dict["resolution_m"] = float(res_m)
            if not card_dict["id"] or not bands or not classes:
                raise ValueError("Fill in an id, the bands and at least one class")
            p = write_card(card_dict)
            all_cards.clear()
            st.success(f"Saved {p}. It is now in the model list.")
        except Exception as e:
            st.error(f"Card not saved: {e}")
