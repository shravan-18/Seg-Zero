# inference_scripts/make_grayscale.py
import argparse
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="path to color image")
    ap.add_argument("--output", required=True, help="path to save grayscale image (e.g., assets/livingroom_gray.png)")
    ap.add_argument("--resize_long", type=int, default=0, help="optional: resize long side (0 = keep size)")
    args = ap.parse_args()

    img = Image.open(args.input).convert("RGB")
    if args.resize_long > 0:
        w, h = img.size
        long_side = max(w, h)
        scale = args.resize_long / float(long_side)
        if scale < 1.0:
            img = img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)

    # Save true grayscale (single-channel “L”)
    gray = img.convert("L")
    gray.save(args.output)
    print(f"Saved grayscale -> {args.output}")

if __name__ == "__main__":
    main()
