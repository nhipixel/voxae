# Voxae-Reason v0 — datasheet

Query/mask pairs for reasoning segmentation on drone imagery. Each sample is
a natural-language query, a symbolic target specification, and the binary
mask that specification resolves to.

## Composition

| Source | Images | Samples | License |
|---|---|---|---|
| VDD | 394 | 2,046 | non-commercial research |
| Semantic Drone Dataset (TU Graz) | 200 | 922 | non-commercial |
| UAVid | 20 | 94 | CC BY-NC-SA 4.0 |
| **Total** | **614** | **3,062** | |

Query families: affordance 2,069, referring 993. Target types: class union
1,594, components 1,468. About 20 distinct semantic classes across the three
source vocabularies.

## How it was built

Masks are never drawn by a model. A language model authors the query text and
picks a symbolic target — a class union (optionally with a near-exclusion
radius or a minimum component area), specific connected components by rank,
or a metric predicate — and the mask is then computed programmatically from
the source dataset's annotation. Every mask is therefore exactly reproducible
from its specification, and a specification that cannot be resolved is
dropped rather than repaired.

Affordance queries are required to carry a constraint: a bare single-class
union is rejected, because the resulting mask would be identical to
"segment the &lt;class&gt;" and the reasoning in the query would be decorative.

Generated with `anthropic/claude-sonnet-5` via prompt version `qgen_v3`,
seed 0. Responses are cached by (prompt version, model, image, seed), so
regeneration is deterministic.

## Quality control

Automatic rules, with per-rule failure accounting (5,199 raw specs -> 3,062
passing; the raw count includes re-emitted cached samples from repeated runs,
all removed by dedupe):

| Rule | Rejects |
|---|---|
| duplicate text within an image | 1,300 |
| mask area outside 0.1-60% | 545 |
| singular reference covering too much of the frame | 271 |
| spatial phrasing with a non-spatial target | 21 |
| trivial affordance target (bare single class) | 0 |

Human review: a 100-sample audit reviewed by the maintainer scored **0.97**
(3 rejects), against a 0.90 threshold. A model-assisted cross-check on a
28-sample stratified subset — weighted toward the riskiest categories
(catch-all class targets, near-exclusion targets, component targets labeled
as affordance) — agreed at ~0.92. The cross-check is not an independent human
label and is reported as a secondary signal only.

The three rejects are informative and reproducible:

- Two were the same shape: a query implying usable size ("ground firm enough
  to bear a heavy truck") whose target was a multi-class union with no area
  floor, so the mask included slivers no vehicle could use. The QC rule
  requiring affordance targets to carry a constraint counts a multi-class
  union as sufficient, which is too permissive when the query's reasoning is
  about size rather than category. 358 samples (11.7%) share this shape:
  multi-class union, no floor, no exclusion, size/usability language, median
  area 27.4%.
- One showed the catch-all class problem directly: a 100px vegetation
  exclusion removed the genuine open ground, leaving mostly building-adjacent
  `other` fragments, so the mask contradicted "open ground clear of
  vegetation".

## Known limitations

- **Catch-all class semantics.** VDD's `other` class backs 8.2% of targets.
  It is genuinely "ground that is not road, roof, wall, vegetation, or
  water", so descriptions like "open terrain" are usually fair, but some
  generated phrasings ("paved or built-up patch") do not match its contents.
  Masks are correct by construction; the language occasionally is not.
- **Family precision.** 23% of affordance samples use component targets and
  read closer to referring queries. The mask matches the text; the family
  label is imprecise.
- **Source imbalance.** VDD contributes 67% of samples and UAVid 3%, because
  UAVid records were capped at 20 images. Models trained on this will see
  VDD's vocabulary most.
- **No metric family.** All three sources ship with EXIF/altitude metadata
  stripped, so ground-sample distance cannot be recovered and no
  metric-predicate queries were generated. The pipeline supports them; the
  data does not.
- **Semantic, not instance, masks.** Touching instances of a class merge into
  one connected component, so a component target can be a fused cluster
  rather than a single object. Oversized singular references are rejected,
  but the boundary is heuristic.

## Distribution

Annotations only. The dataset redistributes no source imagery; each sample
stores a relative path into the original dataset, and the preparation scripts
in this repository rebuild the local layout. Derived annotations inherit the
non-commercial terms of the source datasets and are released under
CC BY-NC-SA 4.0.

One exception, outside the dataset itself: a small number of **UAVid** frames
ship in the demo gallery, so the public demo runs on imagery matching the
training distribution rather than on out-of-distribution stock photography.
UAVid is CC BY-NC-SA 4.0, which permits non-commercial redistribution with
attribution under the same licence; the frames are unmodified and credited in
`voxae/demo/assets/ATTRIBUTION.md`. The Semantic Drone Dataset and VDD forbid
redistribution and appear only as measured results in figures, never as files.
