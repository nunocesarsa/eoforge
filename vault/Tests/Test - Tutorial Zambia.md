---
tags: [test, real-data, zoo]
---
# Test: tutorial data, Zambia

`python tools/get_tutorial_data.py` on your machine, 2026-10-21, with the defaults: 6 x 6 km at -14.95, 28.35 (farms and miombo woodland north of Lusaka), UTM 35S. Guide: [[Tutorial]].

## Download
| Scene | Date | Cloud in the area |
|---|---|---|
| start | 2024-04-23 | 0 % |
| end | 2024-08-26 | 0 % |
| burn | 2024-09-05 | 0 % |
| cloudy | 2024-02-03 | 38 % |

All from Earth Search `sentinel-2-l2a`, tile 35LPD. The darkest pixels were below 900, so the files carried no +1000 offset and none was removed. ESA WorldCover 2021 for the area: grassland 32 %, shrubland 27 %, tree cover 24 %, cropland 17 %, built-up 1 %. Sen1Floods11: two Nigeria chips (1095404, 143329).

## Dynamic World on the end-of-season scene (26 August)
![[tutorial_dynamic_world_dry_season.png]]

Bare 78 %, crops 19 %, shrub 2 %, trees 0.4 %. Against WorldCover (`compare_maps.py`): 52 % of tree cover and 92 % of grassland came out as bare; agreement after matching 41 %.

This is the season, not a bug: in late August the miombo woodland has dropped its leaves and the grass is dry, so the image is brown and the scene really looks bare. Dynamic World labels a single date, while WorldCover is an annual map. In addition the input is L2A while Dynamic World was trained on L1C ([[Sample datasets per model]]). The tutorial now runs land cover on the April (green) scene first and the August scene second, to show that difference.
