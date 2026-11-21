"""
Assemble a demo video from the recorded frames:

    ./.venv/bin/python -m demos.make_video --scene spine [--out demos/out/spine/spine_demo.mp4] [--fps 20]

Inputs (frame index k matches across all of them):
  demos/out/<scene>/slicer/slicer_<k>.png   3D Slicer (cut-open body, B-mode, plot), from slicer_demo.py offline mode
  demos/out/<scene>/pybullet/pb_<k>.png     PyBullet overview, from run_demo.py --record
  demos/out/<scene>/pybullet/cut_<k>.png    PyBullet close-up with the near half of the skin hidden
Layout: Slicer on top, the two PyBullet views below, a caption bar at the bottom. Missing PyBullet frames are
skipped (Slicer-only video).
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPTIONS = {
    "spine": "Lumbar spine (L4), transverse sweep while the probe presses 1-6 N. Patient-specific linear FEM "
             "(fitted fat 3.4 kPa): skin, fat and muscle deform, and the B-mode is resliced through the deformed "
             "tissue, so the vertebra echo moves with the pressure.",
    "muscle": "Paraspinal muscle, sweep over the erector spinae while pressing 1-6 N: the fascia line in the B-mode "
              "moves with the compression.",
    "liver": "Liver, sweep over the right upper abdomen while pressing 1-6 N. The liver (4.1 kPa, Fung 2013) slides "
             "under the abdominal wall (normal-only tie); the bright line in the B-mode is the liver capsule.",
    "vein": "Groin vessels, transverse view while pressing 1-6 N (near-void lumen, fitted fat). In this linear model "
            "the lumen area shrinks only about 7 % at 6 N; real femoral veins collapse fully near 82 mmHg, so the "
            "model under-compresses veins (DEC-062).",
}


def font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def wrap(draw, text, fnt, width):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=fnt) > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = t
    return lines + ([cur] if cur else [])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--src", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--width", type=int, default=1920)
    a = ap.parse_args(argv)
    src = a.src or os.path.join(ROOT, "demos", "out", a.scene)
    out = a.out or os.path.join(src, f"{a.scene}_demo.mp4")
    sl = sorted(glob.glob(os.path.join(src, "slicer", "slicer_*.png")))
    if not sl:
        raise SystemExit(f"no Slicer frames in {src}/slicer")
    W = a.width
    cap_f, tag_f = font(22), font(20)
    tmp = tempfile.mkdtemp(prefix="mtp_video_")
    try:
        for n, path in enumerate(sl):
            k = int(os.path.basename(path)[7:11])
            s = Image.open(path).convert("RGB")
            s = s.resize((W, round(s.height * W / s.width)))
            pbs = [os.path.join(src, "pybullet", f"{p}_{k:04d}.png") for p in ("pb", "cut")]
            pbs = [Image.open(p).convert("RGB") for p in pbs if os.path.exists(p)]
            row_h = 0
            if pbs:
                pw = W // len(pbs)
                pbs = [im.resize((pw, round(im.height * pw / im.width))) for im in pbs]
                row_h = max(im.height for im in pbs)
            d0 = ImageDraw.Draw(s)
            lines = wrap(d0, CAPTIONS.get(a.scene, a.scene), cap_f, W - 40)
            cap_h = 20 + 30 * len(lines)
            H = s.height + row_h + cap_h
            H += H % 2
            fr = Image.new("RGB", (W, H), (14, 16, 22))
            fr.paste(s, (0, 0))
            d = ImageDraw.Draw(fr)
            x = 0
            for im, tag in zip(pbs, ("PyBullet: whole body", "PyBullet: cutaway close-up")):
                fr.paste(im, (x, s.height))
                d.text((x + im.width - 20 - d.textlength(tag, font=tag_f), s.height + 10), tag, font=tag_f,
                       fill=(230, 235, 240))       # top right: the PyBullet HUD sits top left
                x += im.width
            d.text((W - 20 - d.textlength("3D Slicer", font=tag_f), 40), "3D Slicer", font=tag_f, fill=(230, 235, 240))
            y = s.height + row_h + 10
            for ln in lines:
                d.text((20, y), ln, font=cap_f, fill=(220, 225, 232))
                y += 30
            fr.save(os.path.join(tmp, f"f_{n:05d}.png"))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i",
                        os.path.join(tmp, "f_%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        "-preset", "slow", "-movflags", "+faststart", out], check=True)
        print("wrote", out, f"({len(sl)} frames, {os.path.getsize(out) / 1e6:.1f} MB)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
