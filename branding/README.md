# Tercet, the Canto mascot

Tercet is a small orange gecko whose angular tail carries three teal hits, a hit triplet: the unit that Canto chains
into tracks. In the banner it drags a sack of letters into a dark room, climbs a ladder and hangs C, A, N, T, O on a
wire one at a time, then watches a particle track run under the lit word.

| file | what it is |
|---|---|
| `canto_banner.gif` | the animated README banner, 960 x 320, 18 s at 14 frames per second, looping; its first frame is the finished sign |
| `canto_banner_final.png` | the finished sign |
| `canto_mascot.png`, `canto_mascot_front.png` | the mascot alone (side view; sitting, facing the viewer), transparent background, 24x |
| `canto_mascot_spritesheet.png`, `.json` | every pose of the banner on a 24 px grid, 6x, transparent |
| `canto_mascot_concepts.png` | the three silhouettes considered (Tercet, Hop, Helix) |
| `mascot/` | the source: `tercet.py` (the pixel parts), `canto_banner.py` (scene and story), `assets.py` (static assets), `review.py` (contact sheets of frames) |

Embed the banner at the top of a README (the path is relative to the repository root; 768 px is exactly 4x the
192 px canvas, so every pixel stays square):

```html
<p align="center">
  <img src="branding/canto_banner.gif" width="768"
       alt="Tercet, a small orange gecko, hangs the letters C, A, N, T, O on a wire in the dark">
</p>
```

Regenerate everything (Pillow only, a few seconds on a CPU):

```
cd branding/mascot
python canto_banner.py   # canto_banner.gif, canto_banner_final.png, banner_info.json
python assets.py         # concepts, mascot, sprite sheet
```

The banner is drawn on a 192 x 64 canvas and enlarged 5x with nearest-neighbour sampling, with one fixed palette of
under 200 colours for every frame, so the pixels stay crisp and the letters do not shift between frames.
