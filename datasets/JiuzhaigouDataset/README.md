# Jiuzhaigou landslide dataset

1337 co-registered RGB + DEM tiles (256 × 256) with landslide polygons, cut from a
post-event orthophoto mosaic of the 2017 M<sub>s</sub> 7.0 Jiuzhaigou earthquake area
(Sichuan, China).

```
images/            1337 × .jpg   256×256 RGB
dem/               1337 × .tif   256×256 float32 elevation (metres)
masks/             1337 × .png   256×256 uint8 {0, 255}
annotations.json   COCO, 1337 images / 1884 landslide polygons
```

RGB, DEM and mask share the same filename stem, so the three modalities index each
other directly.

**Full dataset card — provenance, both split protocols, and known caveats
(including an unreliable COCO `area` field and a stale `licenses` block) — is at
[`../../docs/JIUZHAIGOU_DATASET.md`](../../docs/JIUZHAIGOU_DATASET.md). Read it before
using the data.**

Split lists live one level up: `../jiuzhaigou_split_*.txt` (random) and
`../jiuzhaigou_geosplit_*.txt` (spatial-block-disjoint).
