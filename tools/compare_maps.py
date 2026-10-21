"""How does a model result agree with a reference map? Prints a cross-table.

    python tools/compare_maps.py tutorial/eoforge_results/<run>/classes.tif tutorial/worldcover_2021.tif

The reference is resampled onto the result's grid (nearest neighbour). Class names come from the
files' tags (class_<value>) when present. Class schemes usually differ, so the table shows, for each
reference class, what share of its pixels got each result class. "Agreement after matching" maps
each result class to the reference class it overlaps most and counts matching pixels: an upper
bound on accuracy, useful to compare models on the same reference, not a formal accuracy.
"""
from __future__ import annotations

import argparse

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


def names(src):
    t = src.tags()
    return {int(k[6:]): v for k, v in t.items() if k.startswith("class_") and k[6:].lstrip("-").isdigit()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("reference")
    ap.add_argument("--min-share", type=float, default=1.0, help="hide cells below this percent")
    a = ap.parse_args(argv)
    with rasterio.open(a.result) as r:
        res = r.read(1).astype("int64")
        rnod = r.nodata
        rn = names(r)
        prof = r.profile
    with rasterio.open(a.reference) as f:
        ref = np.full(res.shape, -9999, "int64")
        reproject(rasterio.band(f, 1), ref, src_transform=f.transform, src_crs=f.crs, src_nodata=f.nodata,
                  dst_transform=prof["transform"], dst_crs=prof["crs"], dst_nodata=-9999, resampling=Resampling.nearest)
        fn = names(f)
        fnod = f.nodata
    ok = ref != -9999
    if fnod is not None:
        ok &= ref != fnod
    ok &= ref >= 0                          # e.g. Sen1Floods11 uses -1 for no data
    if rnod is not None:
        ok &= res != rnod
    if not ok.any():
        raise SystemExit("No overlapping valid pixels")
    rv, fv = res[ok], ref[ok]
    rc, fc = np.unique(rv), np.unique(fv)
    rlab = lambda v: rn.get(int(v), str(v))
    flab = lambda v: fn.get(int(v), str(v))
    print(f"{ok.sum():,} pixels compared\n")
    print("Each reference class: share of its pixels per result class (%)")
    for c in fc:
        m = fv == c
        vals, cnt = np.unique(rv[m], return_counts=True)
        parts = [f"{rlab(v)} {100 * n / m.sum():.0f}" for v, n in sorted(zip(vals, cnt), key=lambda x: -x[1])
                 if 100 * n / m.sum() >= a.min_share]
        print(f"  {flab(c):22s} ({100 * m.mean():4.1f}% of area): " + ", ".join(parts))
    match = 0
    print("\nResult class -> reference class it overlaps most")
    for c in rc:
        m = rv == c
        vals, cnt = np.unique(fv[m], return_counts=True)
        best = vals[cnt.argmax()]
        match += cnt.max()
        print(f"  {rlab(c):22s} -> {flab(best)} ({100 * cnt.max() / m.sum():.0f}% of its pixels)")
    print(f"\nAgreement after matching: {100 * match / ok.sum():.1f}%")
    if set(rc) <= {0, 1} and set(fc) <= {0, 1}:             # binary maps, e.g. water
        tp = int(((rv == 1) & (fv == 1)).sum()); fp = int(((rv == 1) & (fv == 0)).sum()); fnn = int(((rv == 0) & (fv == 1)).sum())
        iou = tp / max(1, tp + fp + fnn)
        print(f"Binary: IoU of class 1 {iou:.3f}, precision {tp / max(1, tp + fp):.3f}, recall {tp / max(1, tp + fnn):.3f}")


if __name__ == "__main__":
    main()
