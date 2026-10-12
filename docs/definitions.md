# Definitions (brief §5d)

Fixed before any performance number is produced. Any later change gets a dated entry in DECISIONS.md with its
reason, and is never made to match a reference number. All numbers are "within this simulation (ACTS generic
detector, no pileup)". Units: mm, GeV, rad.

## 1. Events

- Samples: `ttbar` (ttbar_nopileup_10K), `hnl10_100` (WmuHNL10GeV100mm_nopileup_10K), `hnl15_100`
  (WmuHNL15GeV100mm_nopileup_5K, unlabeled; external test set), `hnl15_200` (WmuHNL15GeV200mm_nopileup_10K).
- Dropped before anything else: the 4 corrupted events in `docs/dropped_events.tsv` (any hit more than 50 mm
  outside its module's padded envelope; DECISIONS.md 2026-10-03). The external 15 GeV 100 mm set has 4,999 events.
- Splits: `splits/*.txt` (`hnl.data.splits`, seed 0): ttbar 80/10/10; `hnl10_100` and `hnl15_200` one joint 80/10/10
  split by event index (event i of both shares its underlying event); `hnl15_100` external only. Selection of
  anything (models, thresholds, working points, bins) uses val only.

## 2. Hits, particles, truth edges

- Hit position: `x` = (r, phi, z) float32 as stored; Cartesian x = r cos(phi), y = r sin(phi) in float64.
- Hit order along a particle: the chain of `layerless_true_edges` (each particle's hits form one chain; verified on
  all events), first hit = the chain head.
- Particle quantities from the table (`particles`): id (int64), PDG, q, pT = hypot(px, py), eta. **The table momenta
  are final-state values** (DECISIONS.md 2026-10-03); for muons they agree with the hits' curvature to 0.2 %.
- Space points of a particle P: all hits with `pid == P`; |P| is their number.
- Pixel hits: volumes 7, 8, 9. First-hit radius: r of the chain head.
- True edges: the npz `layerless_true_edges` (consecutive hits along each particle's chain), mapped to acorn's
  `track_edges` by row index. Every edge metric uses these. Edge counts are in unique unordered hit pairs.
  acorn's own `edge_y` efficiency is reported separately for comparison.

## 3. Truth-hit helix quantities (brief §5b; `hnl.truth.fit_events`)

- Field: uniform B = 2 T along +z (also stated in the Zenodo record). A particle of charge q turns with rotation
  sense sigma = -sign(q) (clockwise seen from +z for q > 0); fixed on W muons (|d0| median 0.0198 mm vs 3.2 mm for
  the other sign). Validated on all events: W-muon |d0| median 0.018-0.019 mm, 90 % below 0.052-0.061 mm in every
  sample; d0 resolution from toy helices 0.014-0.034 mm for |d0| <= 100 mm (RESULTS.md 2026-10-03).
- Fit: robust (Huber) circle fit in float64 of the particle's own hits in the transverse plane, in perigee
  parameters (d0, phi0, kappa = sigma / R), Levenberg-Marquardt on IRLS weights, monotone in the Huber objective.
  Huber scales per volume (transverse / z, mm): pixel 0.05 / 0.10; short-strip barrel 0.10 / 0.60, endcaps
  0.30 / 0.20; long-strip barrel 0.15 / 3.00, endcaps 1.00 / 0.30.
  - **Muons** (|PDG| = 13): fixed R = 1000 pT / (0.3 x 2.0) mm = 1666.67 mm x pT[GeV] (table pT), sigma from q.
  - **All other particles**: free curvature, started from the fixed-R solution (their table pT is the final pT,
    lowered by energy loss: electrons by 4.5 % in median, hadrons by 0.4 %).
  - Particles with fewer than 3 hits get no fit (they are never targets).
- d0: signed transverse distance of the point of closest approach p to the z axis, d0 = (p x t)_z with t the unit
  direction of motion at p. Bins use |d0|.
- z0, cot(theta): robust (Huber) line fit z = z0 + s cot(theta), s = signed transverse arc length from p.
- r_v (HNL daughters only): transverse radius of the two-track vertex of the event's HNL-daughter pair, both
  daughters fitted: the intersection of the two fitted circles (or, if they do not intersect, the midpoint of their
  closest points), choosing the candidate inside min(first-hit radius) + 5 mm, else the one with the smaller z gap.
  A daughter has no r_v if its partner has no fit (< 3 hits or absent from the table) or no candidate is inside;
  such daughters form the r_v category "no vertex" and are counted in every r_v-integrated number.
- Opening angle of the pair: 3D angle between the two daughters' table momenta.

## 4. Categories and targets

- Categories: **HNL daughter** (|parent| = 50), **W muon** (|parent| = 24 and |PDG| = 13), **other**. Labeled
  samples use column 12. In `hnl15_100`: HNL daughters = the opposite-charge generation-0 muon pair that is alone (as
  generation-0 particles) at one vsec > 1, smallest such vsec if several; W muon = the only generation-0 muon at
  vsec = 1 (validated: RESULTS.md 2026-10-03).
- **Target (reconstructable) particle**: pT > 1 GeV, |eta| < 4 (table values) and >= 5 space points (the paper's
  definition). Also reported: the subset with >= 3 pixel hits.
- Truth-hit |d0| bins [mm]: [0, 0.1), [0.1, 1), [1, 3), [3, 10), [10, 30), [30, 100), [100, 300), and >= 300 shown
  as overflow. The |d0| of a particle is the same for every model.
- r_v bins for HNL daughters [mm]: [0, 10), [10, 30), [30, 100), [100, 200), [200, 300), >= 300, plus "no vertex".
- Prompt references: W muons, and ttbar targets with |d0| < 1 mm.
- **Comparison with the paper only** (brief §5f; docs/paper_facts.md): its prompt denominator (4,744 on the external
  set) equals the W muons with >= 5 hits, without the pT / |eta| cuts (113 W muons have table |eta| > 4 and >= 5
  forward hits). For that comparison the prompt denominator is "W muons with >= 5 hits"; the all-track denominator
  is the target definition above (23.3 per event, as the paper's binomial error implies); the displaced one cannot be
  reproduced (the paper's set is about half of the daughters), so all daughter targets and each charge separately are
  shown. These variants are used for that comparison only.

## 5. Matching and track metrics

A reconstructed track T is a set of at least 2 hits (one label per hit from the track builder; unlabelled hits and
isolated hits belong to no track, label -1, as in acorn's connected components). |T| is its number of hits;
|T n P| the number of its hits made by particle P.

- **Double majority (DM)**: T matches P if |T n P| > |T| / 2 and |T n P| > |P| / 2. A particle matches at most one
  track and a track at most one particle.
- **Efficiency**: N(targets with a DM match) / N(targets), per category and bin, as in the paper.
- **Track fake rate**: N(tracks with |T| >= 5 that DM-match no particle) / N(tracks with |T| >= 5).
- **Duplicates**: with single-majority matching (|T n P| > |T| / 2) among tracks with |T| >= 5 (the population of the
  fake rate), a particle with k matched tracks has max(k - 1, 0) duplicates; duplicate rate = sum of duplicates over
  targets / N(targets). (Clarified 2026-10-03 before any result: without the size cut, every 1-2 hit fragment of a
  split track would count as a duplicate.)
- **Pair efficiency**: fraction of events whose two HNL daughters are both targets and both DM-matched.
  **Merge rate**: fraction of such pairs where one track holds at least half of the hits of each daughter.
- **Vertex efficiency** (Stage 1 D): both daughters DM-matched and assigned to one vertex candidate, vs r_v.

## 6. Edge metrics

- Graph size: unique unordered hit pairs per event (acorn's internal doubling of edges is undone before counting).
- Edge efficiency: fraction of true edges (section 2) present in the graph (unordered); edge purity: fraction of
  graph edges that are true edges. Also per layer pair (volume, layer) x (volume, layer).
- Graph-stage ceiling: keep exactly the true edges present in a graph (a perfect classifier), build tracks with
  connected components, and compute the DM efficiency of section 5.
- Working points (pre-registration, docs/preregistration.md): a fixed track fake rate and a fixed edge false-positive
  rate, set on val.
