---
tags: [concept, data, lesson]
---
# Band shift

A model trained on one scene is applied to another (a different year, area or processing level). If a band's values differ a lot, the network gets inputs it never saw.

[[check_band_shift]] runs inside [[predict]] and compares the new scene's band means with the training means, in training standard deviations. Bands beyond ±2 SD trigger a warning.

Found in [[Test - Real Sentinel-2 Slovenia]]: band **B10** (cirrus, an atmospheric band in L1C data) shifted **+6.1 SD** between the 2016 and 2017 composites because its spread within 2017 was tiny. The U-Net's 2016 accuracy dropped to 0.44 while a Random Forest stayed at 0.91. Dropping B01, B09 and B10 (`data.bands`) removed the warning.

The zoo does the same for trained models: the `eoforge` adapter warns in the app and `summary.json` when a band's mean is more than 2 SD from training.

Rule of thumb for land cover: drop atmospheric bands (B01, B09, B10) and prefer surface reflectance (L2A) when transferring between dates.
