Example trained model (EOForge)

A U-Net (base 32, depth 3) trained with eoforge on one LandCoverNet Africa chip, 33LYE_20
(2.56 x 2.56 km, southern Africa, Sentinel-2 L2A 2018 composite, 10 bands B02-B12 without B01/B09).
Classes: water, artificial bare ground, natural bare ground, woody, cultivated, semi-natural vegetation.
Held-out blocks of the same chip: overall accuracy 0.97, kappa 0.93 (see metrics.json).

It shows up in the app as "Trained: example_unet_33LYE_20 (unet)". Try it on
showcase/landcovernet/33LYE_20_s2_2018.tif (its own chip; the dry_ or wet_ bands are matched by name) or other chips;
on other chips expect much weaker results, since it was trained on 2.5 km of labels.

Retrain:  python tools/fetch_landcovernet.py --n 10 --out showcase/landcovernet
          eoforge train runs/example_unet_33LYE_20/config.yaml
(the current download writes wet/dry stacks of 20 bands, so a retrained model uses 20 bands;
this checkpoint was trained on an earlier 10-band single composite.)
