# Navigation losses of Fatras in the no-pileup samples (truth only)

Status: 2026-10-14. This note is descriptive and uses truth only. No model ran and no stored model output was read. For
`generic_iso_v1` and `generic_dense_v1` we read the train and val events only
(`splits/generic_{iso,dense}_{train,val}.txt`). No test event and nothing from the extension `generic_iso_x1` was read.
The truth sidecars hold whole 500-event chunks, and rows of other events were dropped on load. For the Zenodo ttbar
sample we read the train split (`splits/ttbar_train.txt`). All jobs ran on the CPU, on `submit-gpu-express` with the A30
and GTX 1080 nodes excluded. Code: `hnl.truth.nav_losses` (scan) and `hnl.truth.nav_losses_report` (tables). Every
number below is in `results/generated_nav_losses_v1.json`.

## 1. Question and answer

Section 4 of `results/pileup_hp/hp_summary.md` (item 1) reports a gun probe in a constant 2 T field, run on the TrackML
geometry. In that probe, 6.2 % of the gun particles left the world without a log line. The spec-v1 generic samples ran
through the same v15 navigator, also in a constant field, and had not been checked. The H-P decisions of 2026-10-14
(choice 1) added this check. For each particle we ask whether its hits stop before it physically leaves the tracker. We
do the same for the prompt particles of the public Zenodo ttbar sample.

- **Gun particles are lost.** In a constant field the v15 navigator sometimes stops navigating a gun particle. The
  particle then flies on without surfaces, so it gets no more material and no more hits. The loop protection stops it,
  almost always outside the detector. This happens to 6.6 % of the simulated gun particles in `generic_iso` (1,409 of
  21,300) and to 5.2 % in `generic_dense` (3,036 of 57,962). These particles have 1.5-1.8 hits on average, against
  5.2-5.4 for the other gun particles, and about a third have no hit at all. The rate and the hit counts are close to
  those of the TrackML-geometry probe.
- **Premature ends.** After its last hit, a gun particle can still cross a layer at a point covered by a module and
  leave no hit there. We call this a premature end. It affects 7.4 % (iso) and 5.7 % (dense) of the gun particles; about
  70 % of these are the particles that flew out. Most of the rest skip the outer barrel layers or the last disks and
  then leave through the world boundary in the normal way.
- **Holes.** Gun particles also skip layers in the middle of their track: 9.6 % (iso) and 6.4 % (dense) of them have at
  least one hole. Most often it is the first layer after the decay vertex. Per crossing, the hole rate is 4.0 % (iso)
  and 2.2 % (dense), against 0.24 % for prompt particles.
- **Material.** Energy loss below pMin (0.5 GeV) ends 6.3 % (iso) and 1.6 % (dense) of the simulated gun particles.
  These are almost all electrons: bremsstrahlung stops 30-32 % of them. Fatras of both samples has no nuclear
  interactions, so no premature end is hadronic. Navigation losses are the same for every species.
- **Targets.** Without the missing crossings, 596 more gun particles of `generic_iso` would have 5 or more hits and be
  targets. That is 7.0 % of the would-be targets: 250 lose their status through a premature end and 346 through holes
  alone. For the P_G-like population (|d0| >= 10 mm or r_v >= 100 mm), 6,755 targets exist on the simulated hits (train
  and val) and 7,349 would exist without the misses. In `generic_dense`, 984 of 23,610 would-be targets are lost (4.2
  %).
- **The Zenodo ttbar shows the same behaviour.** Our ttbar and the Zenodo ttbar give the same premature-end rate for
  prompt particles (3.65 % and 3.77 %), the same holes per crossing (0.24 % and 0.26 %) and the same pT and |eta|
  profile. In both samples the premature ends are low-pT tracks (pT < 0.5 GeV) at 1 < |eta| < 2.5 that skip their last
  disks. Prompt particles almost never fly out: 13 of 1.31 million in ours and 12 of 0.92 million in Zenodo. There is
  one small difference. Above pT = 1 GeV our ttbar has no premature end, while 0.13 % of such Zenodo particles end
  early, almost all at the outermost barrel layer. Zenodo also has more holes in the central long-strip barrel. The
  Zenodo ttbar holds no displaced particles, so it cannot show the gun-type losses.

## 2. Inputs

| Sample | Events | Particles checked |
|---|---|---|
| `generic_iso_v1`, train + val | 4,810 + 806 | every gun particle (22,464) and every charged prompt ttbar particle with hits |
| `generic_dense_v1`, train + val | 4,780 + 835 | every gun particle (61,642) and every charged prompt ttbar particle with hits |
| Zenodo ttbar, train | 7,999 | every charged prompt particle with hits (924,138) |

A prompt particle is a generator particle of the hard-scatter vertex: barcode secondary vertex 0, generation 0 and
sub-particle 0. We use the same rule in both ttbar samples. The production settings are those of
`docs/new_samples_spec.md` section 2: ACTS v15 Fatras, GenericDetector, constant 2 T along +z, pMin = 0.5 GeV, multiple
scattering, ionisation, bremsstrahlung and photon conversion, and no nuclear interactions.
`results/acts_validation_v1.md` shows that the Zenodo ttbar is reproduced by these settings, including the absence of
any hadronic secondary.

## 3. Method

**Geometry.** We read the sensitive surfaces from `/work/submit/jinboz1/hnl/acts/geometry/detectors.csv`, the v15
GenericDetector. Its geometry ids are those of the Zenodo files. There are 10 barrel layers and 38 disks. The world
boundary is at r = 1028.538 mm and at 3001 mm in |z| (`layer-volumes.csv`). We count a crossing only where a module can
record it. The straight line through the crossing, along the track direction, must meet one of the 8 nearest module
planes inside that module's bounds. The distance inside the bounds is called the depth. This test matters: the -z
short-strip disks (volume 12) of the GenericDetector cover only 96.6-96.7 % of their area, while all other strip disks
cover 100 % (`disk_coverage` in the json). Without the test, crossings through these gaps would count as holes, in both
samples. The pixel disks cover 98.1-98.2 %, because of a 1.7 mm gap between their two rings.

**Two helices per particle.** Both use the constant 2 T field of production.

- *A, from the initial state* (generated samples only). The gun particles take it from the `gun_truth` sidecar, which
  lists every gun particle. The ttbar particles take it from the `truth` sidecar. The expected path ends where Fatras
  ends a particle that is not lost. That is the world boundary or the loop protection of the v15 propagator, whichever
  comes first. The loop protection is `Acts::detail::LoopProtection`, on by default; its path limit is half a turn in
  the transverse plane. A compares the last hit with the outermost layer the particle should reach.
- *B, from the end state* (every sample). The Zenodo files store only the end state of each particle (`hnl.data.npz`),
  so B is the only helix available there. It goes through the stored end position with the end momentum. It starts at
  the point of its circle nearest to the production vertex (generated samples) or nearest to the z axis (Zenodo prompt
  particles). The path ends at the end position, or at the world boundary if the particle ended beyond it. After the
  last hit, B follows the particle's own final trajectory. For a particle that flew on without navigation it is exact,
  because nothing acts on the particle after the loss.

**Counting.** Each hit is matched to the crossing of its own layer nearest to it; a barrel layer can be crossed twice.
The last hit is the matched hit furthest along the helix. A crossing is robust if it lies at least 1 mm (B) or 3 mm (A)
inside a module. It must also lie 10 mm (B) or 20 mm (A) inside the outer edges of its layer, and away from a grazing
crossing of the barrel. For a path that the loop protection ends in A, it must lie 10 mm before the end. A particle has
a **premature end** if at least one robust crossing follows its last hit. A particle without hits has one if its path
has a robust crossing. We also report "2 or more". A **hole** is a robust crossing before the last hit that has no hit.

**End state.** We classify each particle by its end state in the particle table:

- *outside*: more than 1 mm beyond the world boundary, so the particle flew on without navigation;
- *world*: on the world boundary;
- *below_pmin*: p < 0.5 GeV inside the world, so ionisation or bremsstrahlung took it below pMin and Fatras killed it;
- *loop*: inside the world, with the transverse direction turned by about pi, i.e. stopped by the loop protection;
- *inside*: anything else.

B ends at the stopping point. A particle killed by energy loss therefore has a premature end in B only if it skipped a
layer before it was killed. A counts every early end, including the material ones.

## 4. Checks of the method

- **No false early ends where the navigator works.** Prompt particles of our ttbar with pT > 1 GeV show no premature end
  (0 of 360,889). Their hole rate is 0.03 % per crossing, and 0 for pT > 2 GeV and |eta| < 1 (455,483 crossings).
- **A and B agree on the gun.** In `generic_iso` both methods flag 1,367 gun particles, A alone flags 102 and B alone
  59; 17,800 are flagged by neither. For prompt particles, mostly below 0.5 GeV, A drifts because of energy loss, and
  the two disagree more (32,970 / 17,004 / 12,317 / 1,122,546). B is used for every comparison with Zenodo.
- **Unmatched hits.** With B, 0.5 % of prompt particles have a hit on a layer that their helix does not cross: 0.52 % in
  ours and 0.54 % in Zenodo. For gun particles the figure is 1.4 % (iso) and 0.8 % (dense).
- **The Zenodo start point.** For our prompt particles, the distance between the true vertex and the start of B has a
  median of 0.93 mm and a 90th percentile of 7.2 mm. For Zenodo, where the start is the point nearest to the z axis, its
  distance from the axis has a median of 0.95 mm and a 90th percentile of 7.2 mm. The start z lies a median 4.6 mm from
  the event median in ours and 4.7 mm in Zenodo.
- **Lost particles fly to the loop protection.** 97.6 % (iso) and 99.2 % (dense) of the "outside" gun particles end with
  a transverse turn within 0.05 rad of pi. This is where the loop protection stops a particle that flies without
  material.

## 5. Gun particles

### 5.1 How they end

| | generic_iso | generic_dense |
|---|---|---|
| gun particles (train + val) | 22,464 | 61,642 |
| p < 0.5 GeV, not simulated | 1,164 | 3,680 |
| p >= 0.5 GeV but missing from the table | 0 | 0 |
| simulated | 21,300 | 57,962 |
| simulated, with hits | 19,924 | 55,957 |
| end: outside | 6.62 % (1,409/21,300) | 5.24 % (3,036/57,962) |
| end: world | 86.97 % (18,524/21,300) | 93.02 % (53,918/57,962) |
| end: loop | 0.11 % (24/21,300) | 0.14 % (78/57,962) |
| end: below_pmin | 6.30 % (1,342/21,300) | 1.60 % (930/57,962) |
| end: inside | 0.01 % (1/21,300) | 0.00 % (0/57,962) |
| outside: mean hits (others with hits) | 1.543 (5.228) | 1.756 (5.387) |
| outside: without any hit | 37.47 % (528/1,409) | 32.25 % (979/3,036) |
| outside: transverse turn within 0.05 rad of pi | 97.59 % (1,375/1,409) | 99.18 % (3,011/3,036) |
| outside: no robust crossing after the last hit | 27.32 % (385/1,409) | 26.75 % (812/3,036) |

About a quarter of the lost particles were lost after their last robust crossing and miss no hit.

### 5.2 Premature ends

| | generic_iso | generic_dense |
|---|---|---|
| B, >= 1 crossing after the last hit | 7.40 % (1,441/19,476) | 5.74 % (3,222/56,093) |
| B, >= 2 | 4.54 % (884/19,476) | 3.16 % (1,774/56,093) |
| A, >= 1 | 12.00 % (2,461/20,506) | 6.69 % (3,778/56,501) |
| A, >= 2 | 7.61 % (1,560/20,506) | 3.69 % (2,086/56,501) |
| B >= 1, end outside | 1,024 | 2,224 |
| B >= 1, end world | 378 | 931 |
| B >= 1, end loop | 17 | 48 |
| B >= 1, end below_pmin | 22 | 19 |
| A >= 1, end outside | 1,008 | 2,162 |
| A >= 1, end world | 334 | 812 |
| A >= 1, end loop | 15 | 47 |
| A >= 1, end below_pmin | 1,103 | 757 |
| A >= 1, end inside | 1 | 0 |

The denominators are the particles whose helix has at least one robust crossing. A counts the electrons that
bremsstrahlung stops as premature; B does not. The two methods therefore differ mainly by the below_pmin row. The
premature ends that end on the world boundary first miss the long-strip barrel layers most often: 1704 and 1702 account
for 178 of 378 (iso) and 367 of 931 (dense). The last short-strip and long-strip disks come next.

Breakdown, `generic_iso`. The first three columns use the particles with a robust crossing; the hole column uses every
particle with a valid B. pT and |eta| are initial values.

| species | premature end (B) | of which flew out | premature end (A) | mid-track hole (B) |
|---|---|---|---|---|
| e | 7.91 % (220/2,780) | 5.40 % (150/2,780) | 32.24 % (1,267/3,930) | 8.44 % (247/2,927) |
| mu | 7.71 % (311/4,036) | 5.40 % (218/4,036) | 7.48 % (300/4,009) | 10.11 % (420/4,153) |
| pi | 7.29 % (301/4,130) | 4.99 % (206/4,130) | 7.16 % (293/4,094) | 9.61 % (409/4,258) |
| K | 7.50 % (316/4,212) | 5.53 % (233/4,212) | 7.42 % (310/4,179) | 9.33 % (406/4,350) |
| p | 6.79 % (293/4,318) | 5.03 % (217/4,318) | 6.78 % (291/4,294) | 9.97 % (444/4,451) |

| pT [GeV] | premature end (B) | of which flew out | premature end (A) | mid-track hole (B) |
|---|---|---|---|---|
| 0.5-1 | 9.74 % (212/2,177) | 7.21 % (157/2,177) | 16.52 % (386/2,336) | 14.68 % (338/2,302) |
| 1-2 | 8.07 % (239/2,960) | 6.55 % (194/2,960) | 13.68 % (432/3,159) | 13.77 % (428/3,108) |
| 2-5 | 9.74 % (458/4,703) | 8.31 % (391/4,703) | 14.14 % (703/4,971) | 11.46 % (559/4,879) |
| 5-10 | 6.98 % (234/3,354) | 5.25 % (176/3,354) | 10.20 % (355/3,482) | 10.29 % (358/3,479) |
| 10-30 | 2.59 % (83/3,203) | 0.53 % (17/3,203) | 5.38 % (178/3,309) | 2.23 % (72/3,230) |
| >= 30 | 0.00 % (0/2,007) | 0.00 % (0/2,007) | 2.51 % (52/2,071) | 0.35 % (7/2,009) |

| \|eta\| | premature end (B) | of which flew out | premature end (A) | mid-track hole (B) |
|---|---|---|---|---|
| 0-1 | 7.01 % (713/10,172) | 4.60 % (468/10,172) | 8.92 % (923/10,349) | 11.40 % (1,218/10,682) |
| 1-2 | 8.53 % (574/6,726) | 6.33 % (426/6,726) | 15.50 % (1,141/7,360) | 8.90 % (605/6,798) |
| 2-3 | 5.64 % (130/2,304) | 4.60 % (106/2,304) | 13.98 % (352/2,518) | 3.91 % (92/2,353) |
| >= 3 | 8.76 % (24/274) | 8.76 % (24/274) | 16.13 % (45/279) | 3.60 % (11/306) |

| r_v [mm] | premature end (B) | of which flew out | premature end (A) | mid-track hole (B) |
|---|---|---|---|---|
| 0-10 | 2.10 % (5/238) | 0.42 % (1/238) | 6.80 % (17/250) | 0.42 % (1/239) |
| 10-100 | 4.75 % (124/2,608) | 3.34 % (87/2,608) | 10.99 % (306/2,783) | 13.82 % (361/2,613) |
| 100-200 | 8.70 % (238/2,735) | 7.75 % (212/2,735) | 14.46 % (420/2,904) | 11.55 % (320/2,770) |
| 200-300 | 4.23 % (118/2,788) | 2.76 % (77/2,788) | 9.09 % (266/2,926) | 13.44 % (386/2,872) |
| 300-450 | 6.14 % (235/3,829) | 4.49 % (172/3,829) | 10.83 % (437/4,035) | 10.52 % (414/3,934) |
| 450-650 | 10.18 % (455/4,469) | 8.10 % (362/4,469) | 14.06 % (659/4,688) | 6.60 % (309/4,681) |
| 650-800 | 9.47 % (266/2,809) | 4.02 % (113/2,809) | 12.19 % (356/2,920) | 4.46 % (135/3,030) |

Breakdown, `generic_dense`:

| species | premature end (B) | of which flew out | premature end (A) | mid-track hole (B) |
|---|---|---|---|---|
| e | 5.73 % (114/1,989) | 3.77 % (75/1,989) | 28.52 % (778/2,728) | 6.92 % (142/2,053) |
| mu | 5.36 % (146/2,722) | 3.71 % (101/2,722) | 5.15 % (139/2,701) | 5.71 % (159/2,782) |
| pi | 5.91 % (2,333/39,459) | 4.03 % (1,590/39,459) | 5.74 % (2,252/39,219) | 6.57 % (2,641/40,228) |
| K | 5.38 % (475/8,824) | 3.79 % (334/8,824) | 5.23 % (459/8,774) | 6.23 % (560/8,989) |
| p | 4.97 % (154/3,099) | 4.00 % (124/3,099) | 4.87 % (150/3,079) | 5.48 % (173/3,156) |

| pT [GeV] | premature end (B) | of which flew out | premature end (A) | mid-track hole (B) |
|---|---|---|---|---|
| 0.5-1 | 6.93 % (503/7,255) | 5.51 % (400/7,255) | 8.22 % (600/7,296) | 8.79 % (660/7,509) |
| 1-2 | 6.32 % (599/9,472) | 5.80 % (549/9,472) | 7.52 % (718/9,546) | 7.81 % (760/9,734) |
| 2-5 | 6.09 % (830/13,625) | 5.39 % (735/13,625) | 7.06 % (970/13,731) | 7.02 % (975/13,895) |
| 5-10 | 3.79 % (337/8,892) | 2.78 % (247/8,892) | 4.46 % (399/8,956) | 6.67 % (599/8,975) |
| 10-30 | 2.73 % (234/8,569) | 0.69 % (59/8,569) | 3.39 % (293/8,638) | 2.30 % (198/8,615) |
| >= 30 | 0.00 % (0/3,789) | 0.00 % (0/3,789) | 0.29 % (11/3,805) | 0.47 % (18/3,790) |

| \|eta\| | premature end (B) | of which flew out | premature end (A) | mid-track hole (B) |
|---|---|---|---|---|
| 0-1 | 4.17 % (1,075/25,796) | 2.43 % (628/25,796) | 4.53 % (1,168/25,770) | 6.81 % (1,796/26,372) |
| 1-2 | 7.64 % (1,636/21,402) | 5.49 % (1,174/21,402) | 8.92 % (1,944/21,785) | 7.27 % (1,572/21,611) |
| 2-3 | 5.84 % (457/7,829) | 4.76 % (373/7,829) | 7.64 % (604/7,905) | 3.51 % (283/8,056) |
| >= 3 | 5.07 % (54/1,066) | 4.60 % (49/1,066) | 5.96 % (62/1,041) | 2.05 % (24/1,169) |

| r_v [mm] | premature end (B) | of which flew out | premature end (A) | mid-track hole (B) |
|---|---|---|---|---|
| 0-10 | 2.71 % (19/701) | 1.00 % (7/701) | 3.94 % (28/711) | 0.71 % (5/702) |
| 10-100 | 3.53 % (265/7,504) | 2.31 % (173/7,504) | 5.08 % (387/7,620) | 8.63 % (648/7,511) |
| 100-200 | 7.89 % (611/7,748) | 6.70 % (519/7,748) | 8.71 % (679/7,794) | 8.03 % (628/7,818) |
| 200-300 | 2.97 % (240/8,090) | 1.77 % (143/8,090) | 4.09 % (334/8,160) | 9.87 % (820/8,308) |
| 300-450 | 4.40 % (493/11,201) | 2.82 % (316/11,201) | 5.30 % (598/11,291) | 7.05 % (800/11,339) |
| 450-650 | 7.94 % (1,037/13,058) | 6.05 % (790/13,058) | 8.79 % (1,155/13,134) | 4.26 % (568/13,338) |
| 650-800 | 7.15 % (557/7,791) | 3.54 % (276/7,791) | 7.66 % (597/7,791) | 2.52 % (206/8,192) |

In both components the flew-out share is flat across species. It falls above pT = 10 GeV and is zero above 30 GeV. Its
dependence on r_v follows the detector volumes. It is highest for decays at 100-200 mm, near the outer pixel boundary at
182 mm, and at 450-650 mm, below the short-strip volume boundary at 707 mm. Decays within 10 mm of the beam line are
rarely lost (1 of 238 in iso).

### 5.3 Holes

| | gun, iso | gun, dense | prompt, our ttbar | prompt, Zenodo |
|---|---|---|---|---|
| per robust crossing | 4.01 % (3,336/83,098) | 2.18 % (5,245/241,016) | 0.24 % (24,910/10,150,699) | 0.26 % (18,619/7,143,098) |
| particles with >= 1 hole | 9.56 % (1,926/20,139) | 6.42 % (3,675/57,208) | 1.84 % (24,158/1,310,642) | 1.96 % (18,102/923,472) |

Holes per robust crossing by subsystem:

| | gun, iso | gun, dense | prompt, our ttbar | prompt, Zenodo |
|---|---|---|---|---|
| pixel barrel | 11.10 % (936/8,435) | 5.49 % (1,270/23,123) | 0.28 % (7,686/2,715,020) | 0.30 % (5,670/1,910,305) |
| pixel disks | 3.62 % (115/3,181) | 1.90 % (194/10,204) | 0.23 % (6,923/2,940,078) | 0.23 % (4,696/2,072,443) |
| short-strip barrel | 6.88 % (1,984/28,838) | 4.05 % (3,211/79,223) | 0.16 % (2,751/1,689,883) | 0.21 % (2,464/1,189,911) |
| short-strip disks | 0.61 % (105/17,117) | 0.51 % (278/54,571) | 0.40 % (7,519/1,903,949) | 0.41 % (5,424/1,336,107) |
| long-strip barrel | 1.18 % (194/16,434) | 0.64 % (284/44,628) | 0.01 % (28/437,517) | 0.12 % (364/307,058) |
| long-strip disks | 0.02 % (2/9,093) | 0.03 % (8/29,267) | 0.00 % (3/464,252) | 0.00 % (1/327,274) |

In two thirds of the gun particles with a hole (1,308 of 1,926 in iso; 2,448 of 3,675 in dense), the first hole is the
first robust crossing after the decay vertex. For these first-layer holes in iso, the vertex lies a median 40 mm inside
the missed barrel layer, and only 8 % of them lie within 5 mm of it. These are therefore skips of a layer that lies well
beyond the vertex. The holed layers are the short-strip barrel (1302-1308) and the pixel barrel (804-808). Gun particles
that start in the barrel volumes often miss the next barrel layer.

### 5.4 Material and hadronic expectation

- Energy loss below pMin stops 32.0 % of the simulated gun electrons in iso and 30.0 % in dense. It stops 0.20 % and
  0.17 % of the other species. These hadrons and muons start just above threshold: their median initial p is 0.51 GeV,
  and 90 % start below 0.59 GeV (iso) or 0.70 GeV (dense). This is the material expectation, and A counts it (below_pmin
  row of section 5.2). With B, only 22 (iso) and 19 (dense) of these particles also skipped a layer before they were
  killed.
- Fatras of the generic samples has no nuclear interactions (`docs/new_samples_spec.md` section 2). The Zenodo ttbar has
  no hadronic secondaries either: every secondary with hits is an electron (`results/acts_validation_v1.md`). No
  premature end is therefore hadronic. The navigation losses are flat across e, mu, pi, K and p (section 5.2), as
  expected for a navigator effect. In a full simulation, hadronic interactions would add physical early ends to the
  hadrons; these samples contain none.

### 5.5 The >= 5-hit targets

Target: pT > 1 GeV, |eta| < 4 (table values) and >= 5 hits (`docs/definitions.md` section 4). The "would-be" count adds
the missing crossings to each particle's hits: robust holes, plus the crossings after the last hit that lie on a module.
B is used where it is valid and A elsewhere. A particle killed by energy loss has a table pT below 0.5 GeV and is never
counted.

| | iso, all gun | iso, P_G-like | dense, all gun | dense, P_G-like |
|---|---|---|---|---|
| targets | 7,947 | 6,755 | 22,626 | 18,870 |
| targets if no crossing were missed | 8,543 | 7,349 | 23,610 | 19,850 |
| lost: premature end | 250 | 248 | 417 | 413 |
| lost: holes only | 346 | 346 | 567 | 567 |
| lost, share of would-be targets | 6.98 % (596/8,543) | 8.08 % (594/7,349) | 4.17 % (984/23,610) | 4.94 % (980/19,850) |
| targets that miss >= 1 hit | 708 | 694 | 1,292 | 1,270 |
| targets with a premature end | 46 | 46 | 109 | 109 |

P_G-like is |d0| >= 10 mm or r_v >= 100 mm, the population rule of addendum G T1. The target counts equal those of
`results/addendum_g_t1_v1.md` (iso: 6,836 + 1,111 targets and 5,799 + 956 P_G, train + val). By split, the would-be P_G
is 6,309 (train) and 1,040 (val). Almost every lost target is P_G-like (594 of 596 in iso, 980 of 984 in dense).
Separately, 708 iso targets (8.9 %) keep their status but miss at least one hit.

## 6. Prompt particles: our ttbar and the Zenodo ttbar

| | our ttbar | Zenodo ttbar |
|---|---|---|
| events | 11,231 | 7,999 |
| charged prompt particles with hits | 1,311,611 | 924,138 |
| end: outside | 0.00 % (13/1,311,611) | 0.00 % (12/924,138) |
| end: world | 97.60 % (1,280,172/1,311,611) | 97.56 % (901,583/924,138) |
| end: loop | 1.20 % (15,740/1,311,611) | 1.21 % (11,153/924,138) |
| end: below_pmin | 1.19 % (15,654/1,311,611) | 1.23 % (11,361/924,138) |
| premature end, B >= 1 | 3.65 % (45,287/1,241,562) | 3.77 % (32,956/874,639) |
| premature end, B >= 2 | 0.67 % (8,287/1,241,562) | 0.68 % (5,909/874,639) |
| B >= 1, end world | 35,613 | 26,066 |
| B >= 1, end loop | 9,342 | 6,619 |
| B >= 1, end below_pmin | 300 | 241 |
| B >= 1, end outside or inside | 32 | 30 |
| loop-ended particles with a premature end | 59.44 % (9,342/15,718) | 59.46 % (6,619/11,132) |
| pT > 1 GeV: premature end | 0.00 % (0/360,889) | 0.13 % (332/247,518) |
| pT > 1 GeV: holes per crossing | 0.03 % (897/3,161,002) | 0.07 % (1,477/2,169,715) |
| targets with a premature end | 0.00 % (0/346,273) | 0.14 % (332/237,485) |
| every charged prompt particle in the table: end outside | 0.01 % (263/1,742,498) | 0.02 % (199/1,227,251) |

Premature end and holes by pT and |eta| (final-state values, B):

| pT [GeV] | premature end, ours | premature end, Zenodo | >= 2, ours | >= 2, Zenodo | hole, ours | hole, Zenodo |
|---|---|---|---|---|---|---|
| 0-0.3 | 11.76 % (27,821/236,657) | 11.75 % (19,933/169,672) | 3.21 % | 3.18 % | 5.32 % | 5.28 % |
| 0.3-0.5 | 6.57 % (17,006/258,761) | 6.67 % (12,378/185,726) | 0.27 % | 0.28 % | 2.43 % | 2.40 % |
| 0.5-1 | 0.12 % (460/385,255) | 0.12 % (313/271,723) | 0.00 % | 0.00 % | 0.69 % | 0.73 % |
| 1-2 | 0.00 % (0/224,172) | 0.08 % (120/152,879) | 0.00 % | 0.00 % | 0.31 % | 0.59 % |
| 2-5 | 0.00 % (0/103,305) | 0.19 % (138/71,359) | 0.00 % | 0.00 % | 0.14 % | 0.47 % |
| >= 5 | 0.00 % (0/33,412) | 0.32 % (74/23,280) | 0.00 % | 0.01 % | 0.04 % | 0.83 % |

| \|eta\| | premature end, ours | premature end, Zenodo | >= 2, ours | >= 2, Zenodo | hole, ours | hole, Zenodo |
|---|---|---|---|---|---|---|
| 0-1 | 0.03 % (59/232,658) | 0.23 % (374/163,921) | 0.00 % | 0.00 % | 0.01 % | 0.65 % |
| 1-2 | 12.13 % (37,459/308,763) | 12.35 % (26,860/217,485) | 2.68 % | 2.71 % | 3.38 % | 3.44 % |
| 2-2.5 | 4.61 % (7,651/165,967) | 4.86 % (5,637/115,886) | 0.01 % | 0.00 % | 2.26 % | 2.30 % |
| 2.5-3 | 0.07 % (106/161,293) | 0.07 % (77/113,711) | 0.00 % | 0.00 % | 4.17 % | 4.13 % |
| >= 3 | 0.00 % (12/372,881) | 0.00 % (8/263,636) | 0.00 % | 0.00 % | 0.72 % | 0.71 % |

The two ttbar samples match bin by bin wherever pT < 1 GeV or |eta| > 1. There, the premature ends are low-pT endcap
tracks. The layers missed first are the last short-strip disks (1202 and 1412, 1204 and 1410) and the long-strip disks,
with similar counts on the two sides in both samples. Most of these particles still leave through the world boundary,
and most of the others stop at the loop protection. 59 % of all loop-ended particles skipped at least one layer before
that point.

Zenodo differs from ours in the central barrel at high pT. 324 of its 332 premature ends above pT = 1 GeV miss the
outermost barrel layer (1704). It also has more holes per crossing in the long-strip barrel (0.12 % against 0.01 %), and
0.65 % of its central particles have a hole, against 0.01 % in ours. The Zenodo sample may come from a different ACTS
version; these files do not tell. The difference is small (0.14 % of Zenodo prompt targets).

## 7. Reading

- In a constant field, a particle that the v15 navigator loses keeps the hits it made before the loss and stays in the
  sample. It is not removed, and nothing is logged. In `generic_iso` and `generic_dense` this affects 5-7 % of the
  simulated gun particles and almost no prompt particle. With the field map of the pileup build, the same navigator
  state ends in an error, and Fatras removes the particle (2.1 % of the gun probe,
  `results/pileup_hp/f1_containment_v1.md`).
- The losses are a property of a displaced start: decays inside the barrel volumes, away from the beam line. Per
  crossing, gun particles also have 9-16 times more holes than prompt particles, most often at the first layer after the
  decay vertex. Both effects are independent of species and fall at high pT.
- Addendum G defines its populations on the simulated hits, as H-P choice 1 does for the pileup samples. Under that rule
  P_G (`generic_iso`, train + val) holds 6,755 targets. Without the missing crossings it would hold 7,349. The 594 gun
  particles in between are displaced tracks that the simulation truncated below 5 hits.
- For prompt particles, the public Zenodo ttbar and our ttbar behave the same way, apart from the small high-pT barrel
  difference of section 6. Both ttbar samples therefore contain the same low-pT endcap skips, and in both the prompt
  targets (pT > 1 GeV) are almost untouched. The Zenodo HNL samples, whose daughters are displaced, were not part of
  this check.

## 8. Files and jobs

| Path | Content |
|---|---|
| `src/hnl/truth/nav_losses.py` | geometry, module coverage, helices A and B, matching, end categories, scan (sha256 in the json, `code_sha256`) |
| `src/hnl/truth/nav_losses_report.py` | tables and `results/generated_nav_losses_v1.json` |
| `slurm/acts/nav_losses.sbatch` | one scan per sample, or the summary |
| `tests/test_nav_losses.py` | unit tests: crossings, field sign, loop protection, restart of the end-state helix, counting, end categories, module depth |
| `derived/nav_losses_v1/{generic_iso,generic_dense,ttbar}.parquet` | one row per checked particle (and a census row per charged prompt particle in the table), on /ceph under `hnl_lrt` |
| `derived/nav_losses_v1/*_layers.csv` | robust crossings, holes and crossings after the last hit, per layer, kind and method |

Jobs: scans 6820123 (`generic_iso`), 6820124 (`generic_dense`) and 6820125 (Zenodo ttbar); summary 6820135. Earlier
scans without the module-coverage test (6820098-6820100, 6820113) were replaced; one of them (6820098) stopped on an
empty event before writing anything. Each scan took 1-7 minutes on 8 cores.
