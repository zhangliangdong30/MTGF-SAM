# Licensing — read before publishing

**This repository has no top-level `LICENSE` file yet.** That is deliberate: choosing
one is your call, and there are three separate things to decide.

## 1. `sam3_dem/` — already decided

A modified fork of Meta's SAM3, distributed under the **SAM License**
(`sam3_dem/LICENSE`, last updated 2025-11-19). It stays under that licence; you cannot
relicense it. Meta's copyright headers are retained in every file, and the changes are
documented in `sam3_dem/MODIFICATIONS.md`.

Practical consequences to check against the licence text before release:

- the SAM License carries acceptable-use and attribution terms that flow to
  downstream users of `sam3_dem/`;
- **SAM3 base weights are not redistributed here** — `sam3_weights/` is gitignored and
  users must obtain the weights from Meta under the same licence. Keep it that way;
- if you publish your own trained MTGF-SAM checkpoints, they are derived from SAM3
  weights, so the SAM License terms follow them. Say so wherever you host them.

## 2. Your own code — undecided

`src/`, `configs/` and `docs/` are your work. Common choices for research code are
MIT, Apache-2.0 or BSD-3-Clause. Apache-2.0 is the usual pick when the repository sits
next to a permissively-but-not-freely licensed dependency, because it is explicit
about patent grants and about stating modifications.

Whatever you pick, add it as `LICENSE` and note in `README.md` that `sam3_dem/` is
excluded and governed by the SAM License instead.

## 3. The Jiuzhaigou dataset — needs a decision *and* a correction

`datasets/JiuzhaigouDataset/annotations.json` carries a boilerplate `licenses` block
left by the annotation tool:

```json
"licenses": [{"id": 1,
              "name": "Attribution-NonCommercial-ShareAlike License",
              "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"}]
```

That is a **CC BY-NC-SA 2.0** declaration that was almost certainly not chosen
deliberately — the same block also says `"description": "Example Dataset"`,
`"contributor": "Black Jack"` and `"year": 2019`, none of which describe this dataset.

Two things to do before publishing:

1. **Decide the dataset licence.** CC BY-NC-SA 2.0 forbids commercial use and is
   generally discouraged for research datasets (it is also a superseded version;
   CC BY-NC-SA 4.0 or CC BY 4.0 are the current equivalents). CC BY 4.0 is the common
   choice for openly published remote-sensing benchmarks.
2. **Fix the metadata** so `info` and `licenses` state the real provenance,
   contributor and licence. Leaving stale boilerplate in a published dataset creates a
   licence ambiguity that is hard to undo once people have copies.

Also confirm you hold the rights to redistribute the **source imagery and DEM** the
tiles were cut from. Annotations you created are yours; the underlying orthophoto and
elevation data may carry their own provider terms.

## 4. Third-party datasets — not included, and should stay that way

Bijie and Landslide4Sense are referenced by the code but are **not** redistributed
here (both are gitignored). Users fetch them from their original sources under those
sources' terms. Keep it that way.

## Suggested minimum before you hit publish

- [ ] add a top-level `LICENSE` for your own code
- [ ] state in `README.md` that `sam3_dem/` is under the SAM License
- [ ] decide the Jiuzhaigou dataset licence and write it into
      `docs/JIUZHAIGOU_DATASET.md` **and** `annotations.json`
- [ ] fix the stale `info` / `licenses` block in `annotations.json`
- [ ] confirm redistribution rights for the source imagery and DEM
- [ ] confirm `sam3_weights/` and `checkpoints/*.pt` are not staged
