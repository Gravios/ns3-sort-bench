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
| ns3sb CLI (doc page) | a prompt above the summary table it prints | `cli.svg` |
| Paired recordings (doc page) | a juxtacellular trace (orange, large positive spikes) over an extracellular one (white, small troughs), cyan links joining matched spikes | `paired-recordings.svg` |
| Adapter contracts (doc page) | a spec sheet with a tick | `contracts.svg` |
| Installation (doc page) | a download arrow into a tray | `install.svg` |
| Compare (banner chip) | a confusion matrix: the diagonal solid in cyan / white / orange, a little leakage off it | `comparison.svg` |
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
- `doc-<page>-banner.png` (2560 × 640) and `doc-<page>-banner-narrow.png`
  (1600 × 600) for `cli`, `paired-recordings`, `contracts` and `install`:
  the top of each documentation page, the narrow one replacing it below
  600 px.  The page's icon, an "ns3-sort-bench" kicker line, the title and a
  tagline — the same layout as neurosuite-3's `doc/<program>/banner.png`;
- `icons/hi<size>-app-ns3-sort-bench.png` (16, 22, 32, 48, 64, 128, 256 px)
  and `icons/app-ns3-sort-bench.ico` (16, 24, 32, 48, 64, 256 px): the app
  icon at the hicolor theme sizes and for Windows, each size rendered from
  the SVG rather than downscaled.

Banners are dithered to 256 colours with `pngquant` when it is installed
(about 130-240 KB each).

After changing an SVG, run `python3 docs/images/render.py` and look at
`icons/hi16-app-ns3-sort-bench.png` and the chips in `banner.png`: below
32 px only the motif's colours and outline survive.
