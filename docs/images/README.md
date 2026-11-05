# Icons and banners

Same family as neurosuite-3's (see `doc/images/README.md` there): a rounded
navy tile, and a cyan recording trace that runs off both edges of the tile
with one extracellular spike in it.  Each icon puts its own motif above the
trace.

| Icon | Motif | Source |
|---|---|---|
| ns3-sort-bench | three ranked bars (cyan < white < orange), each in a faint column marking the ceiling | `ns3-sort-bench.svg` |
| Sorters (banner chip) | a plug — every sorter plugs in through an adapter | `sorters.svg` |
| Ground truth (banner chip) | a target | `ground-truth.svg` |
| Klusters (banner chip) | copied from neurosuite-3 `src/klusters/src/app-klusters.svg` | `klusters.svg` |
| neurosuite-3 (banner kicker) | copied from neurosuite-3 `doc/images/neurosuite-3.svg` | `neurosuite-3.svg` |

Palette: tile gradient `#0E2236` → `#173F5C`, cyan `#43C8E2`, white
`#EEF4F8`, orange `#FF8F45`.  The SVGs are 512 × 512, shapes and strokes
only (no fonts).

Rendered from them by `render.py` (Chromium via Playwright, Inter font):

- `banner.png` (2560 × 800, shown at half size): the top of the README;
- `banner-narrow.png` (1600 × 680: icon, name, a short line): replaces it
  below 600 px;
- `social-preview.png` (1280 × 640): GitHub's link preview, uploaded by hand
  under Settings → General → Social preview;
- `icon-256.png`.

After changing an SVG, run `python3 docs/images/render.py` and look at the
chips in `banner.png`: at 54 px only the motif's colours and outline survive.
