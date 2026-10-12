"""Contact sheets of the banner frames for review: python review.py OUT.png [start] [stop] [step] [zoom]"""
import sys
from PIL import Image, ImageDraw
import canto_banner as B

d = B.Director()
states = d.story()
a = int(sys.argv[2]) if len(sys.argv) > 2 else 0
b = int(sys.argv[3]) if len(sys.argv) > 3 else len(states)
step = int(sys.argv[4]) if len(sys.argv) > 4 else 6
z = int(sys.argv[5]) if len(sys.argv) > 5 else 2
idx = list(range(a, min(b, len(states)), step))
cols = 4
w, h = B.NW * z, B.NH * z + 12
sheet = Image.new("RGB", (cols * (w + 4), ((len(idx) + cols - 1) // cols) * (h + 4)), (60, 60, 60))
dr = ImageDraw.Draw(sheet)
for n, f in enumerate(idx):
    im = B.render(states[f], f).convert("RGB").resize((w, B.NH * z), Image.NEAREST)
    x, y = (n % cols) * (w + 4), (n // cols) * (h + 4)
    sheet.paste(im, (x, y + 12))
    dr.text((x + 2, y), f"f{f}", fill=(255, 255, 255))
sheet.save(sys.argv[1])
print(len(states), "frames; hero", d.hero_index)
