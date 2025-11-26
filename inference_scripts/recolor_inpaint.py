"""
Ground & Colorize Pipeline (Seg-Zero + SDXL Inpainting + LAB fallback)

Usage (example):
python inference_scripts/ground_and_colorize.py \
  --image_path assets/grayscale.png \
  --tasks_json inference_scripts/tasks.json \
  --final_out inference_scripts/final_colorized.png \
  --save_debug

The tasks.json should be a list of entries like:
[
  {"text": "the hat worn by the lady", "color": "hotpink"},
  {"text": "the lady's jacket",        "color": "navy"},
  {"text": "the umbrella",             "color": "#00b3b3"}
]

Notes:
- White = edit region, black = keep (mask is produced automatically).
- SDXL tries to color first; if it returns gray, LAB fallback applies target color deterministically.
- You can adjust guidance/strength/steps globally; per-target knobs can also be added if needed.
"""

import os
import json
import re
from skimage import color
import argparse
import numpy as np
import torch
from PIL import Image as PILImage, ImageFilter, ImageOps, Image
import matplotlib.pyplot as plt
import cv2

# Seg-Zero / Qwen / SAM2
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Diffusion (SDXL inpainting)
from diffusers import AutoPipelineForInpainting
from skimage import color

# ------- Named colors for fallback -------
NAMED_RGB = {
    "hotpink": (255, 105, 180),
    "pink": (255, 192, 203),
    "red": (220, 20, 60),
    "scarlet": (255, 36, 0),
    "orange": (255, 140, 0),
    "yellow": (255, 215, 0),
    "green": (34, 139, 34),
    "teal": (0, 128, 128),
    "cyan": (0, 255, 255),
    "blue": (65, 105, 225),
    "navy": (0, 0, 128),
    "purple": (138, 43, 226),
    "violet": (238, 130, 238),
    "brown": (139, 69, 19),
    "beige": (245, 245, 220),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "grey": (128, 128, 128),
}

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser()
    # I/O
    p.add_argument("--image_path", type=str, default="assets/grayscale.png")
    p.add_argument("--tasks_json", type=str, required=True,
                   help="JSON file: list of {'text':..., 'color':...} (color name or #RRGGBB)")
    p.add_argument("--final_out", type=str, default="inference_scripts/final_colorized.png")

    # Where to save per-target outputs/masks/overlays
    p.add_argument("--out_dir", type=str, default="inference_scripts")
    p.add_argument("--save_debug", action="store_true")

    # Seg-Zero models
    p.add_argument("--reasoning_model_path", type=str, default="pretrained_models/VisionReasoner-7B")
    p.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")

    # SDXL model
    p.add_argument("--sdxl_model_id", type=str, default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")

    # Diffusion knobs (global)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=6.0)
    p.add_argument("--strength", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--disable_safety", action="store_true")

    # Mask post-proc
    p.add_argument("--erode", type=int, default=0)
    p.add_argument("--dilate", type=int, default=0)
    p.add_argument("--feather", type=float, default=1.2)
    p.add_argument("--invert_mask", action="store_true", help="invert all produced masks before coloring")

    # Resize policy
    p.add_argument("--keep_original_size", action="store_true")

    # LAB fallback
    p.add_argument("--fallback_alpha", type=float, default=0.8,
                   help="Blend strength for deterministic color if diffusion stays gray (0..1)")

    return p.parse_args()

# ---------------- utility -------------------
def ensure_dir(path: str):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)

def parse_color(spec: str):
    spec = spec.strip().lower()
    if spec.startswith("#") and len(spec) == 7:
        r = int(spec[1:3], 16); g = int(spec[3:5], 16); b = int(spec[5:7], 16)
        return (r, g, b)
    return NAMED_RGB.get(spec, NAMED_RGB["hotpink"])

def tidy_mask_bool(mask_bool, erode_px=0, dilate_px=0, feather_sigma=0.0):
    m = (mask_bool.astype(np.uint8)) * 255
    if erode_px > 0:
        k = 2*erode_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.erode(m, kernel)
    if dilate_px > 0:
        k = 2*dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.dilate(m, kernel)
    if feather_sigma and feather_sigma > 0:
        ksize = max(1, int(6*feather_sigma) | 1)
        m = cv2.GaussianBlur(m, (ksize, ksize), feather_sigma)
        m = np.clip(m, 0, 255).astype(np.uint8)
    return m

def overlay_and_save(base_img: PILImage.Image, mask_bool: np.ndarray, save_path: str, title: str):
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1); plt.imshow(base_img); plt.axis("off"); plt.title("Original")
    plt.subplot(1,2,2)
    plt.imshow(base_img, alpha=0.6)
    rgb = np.zeros((base_img.height, base_img.width, 3), dtype=np.uint8)
    rgb[mask_bool] = [255,0,0]
    plt.imshow(rgb, alpha=0.4); plt.axis("off"); plt.title(title)
    ensure_dir(save_path)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

# ------------- Seg-Zero parsing -------------
def extract_bbox_points_think(output_text, x_factor, y_factor):
    json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
    pred_bboxes, pred_points = [], []
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            for item in data:
                bx = [
                    int(item['bbox_2d'][0] * x_factor + 0.5),
                    int(item['bbox_2d'][1] * y_factor + 0.5),
                    int(item['bbox_2d'][2] * x_factor + 0.5),
                    int(item['bbox_2d'][3] * y_factor + 0.5),
                ]
                pt = [
                    int(item['point_2d'][0] * x_factor + 0.5),
                    int(item['point_2d'][1] * y_factor + 0.5),
                ]
                pred_bboxes.append(bx); pred_points.append(pt)
        except Exception:
            pred_bboxes, pred_points = [], []

    think_match = re.search(r'<think>(.*?)</think>', output_text, re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else ""
    return pred_bboxes, pred_points, think_text

# ------------- Grounding (Seg-Zero) ---------
def build_reasoning_inputs(processor, img_pil, question_text, resize_size=840):
    Q_TEMPLATE = (
        "Please find \"{Question}\" with bboxs and points."
        "Compare the difference between object(s) and find the most closely matched object(s)."
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
        "Output the bbox(es) and point(s) inside the interested object(s) in JSON format."
        "i.e., <think> thinking process here </think>"
        "<answer>{Answer}</answer>"
    )
    message = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img_pil.resize((resize_size, resize_size), PILImage.BILINEAR)},
            {"type": "text",
             "text": Q_TEMPLATE.format(
                 Question=question_text.lower().strip("."),
                 Answer=(
                    "[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}, "
                    "{\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410]}]"
                 )
             )}
        ]
    }]
    text = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)]
    image_inputs, video_inputs = process_vision_info([message])
    inputs = processor(
        text=text, images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    )
    return inputs

def segzero_ground(
    reasoning_model, processor, seg_model,
    image_pil: PILImage.Image, query_text: str,
    erode=0, dilate=0, feather=1.2,
):
    """Return boolean HxW mask for the grounded region (union of all predicted objects)."""
    original_w, original_h = image_pil.size
    resize_size = 840
    x_factor, y_factor = original_w/resize_size, original_h/resize_size

    inputs = build_reasoning_inputs(processor, image_pil, query_text, resize_size=resize_size).to("cuda")

    generated_ids = reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    bboxes, points, think = extract_bbox_points_think(output_text, x_factor, y_factor)
    print(f"[ground] '{query_text}' → {len(points)} point(s). Think: {think[:120]}...")

    mask_all = np.zeros((image_pil.height, image_pil.width), dtype=bool)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        seg_model.set_image(image_pil)
        for bbox, point in zip(bboxes, points):
            masks, scores, _ = seg_model.predict(point_coords=[point], point_labels=[1], box=bbox)
            mask = masks[np.argsort(scores)[::-1]][0].astype(bool)
            mask_all = np.logical_or(mask_all, mask)

    # Clean / feather for diffusion
    mask_u8 = tidy_mask_bool(mask_all, erode_px=erode, dilate_px=dilate, feather_sigma=feather)
    if mask_u8.max() > 0:
        mask_bool = mask_u8 > 127
    else:
        mask_bool = mask_all

    return mask_bool

# --------- SDXL + LAB fallback coloring ----------
def sdxl_colorize_one(
    pipe, base_pil, mask_bool, target_color,
    steps=50, guidance=6.0, strength=0.5,  # note: lower default
    seed=1234, disable_safety=False, keep_original_size=False,
    save_debug=False, debug_prefix="debug",
):
    # --- build soft L mask and a pre-colored hint image ---
    mask_L = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")
    if save_debug: mask_L.save(f"{debug_prefix}_mask.png")

    rgb = np.array(parse_color(target_color), dtype=np.float32) / 255.0
    base_np = np.asarray(base_pil).astype(np.float32) / 255.0
    m = (np.asarray(mask_L).astype(np.float32) / 255.0)[..., None]

    lab_base = color.rgb2lab(base_np)
    tgt_lab  = color.rgb2lab(rgb.reshape(1,1,3)).reshape(1,1,3)
    L = lab_base[..., :1]
    ab_hint = tgt_lab[..., 1:3]

    # blend a *subtle* hint (alpha ~0.5) inside the mask
    alpha_hint = 0.5
    ab = lab_base[..., 1:3] * (1 - alpha_hint*m) + ab_hint * (alpha_hint*m)
    lab_hint = np.concatenate([L, ab], axis=-1)
    hint_rgb = np.clip(color.lab2rgb(lab_hint), 0, 1)
    hint_img = Image.fromarray((hint_rgb * 255).astype(np.uint8))

    if save_debug: hint_img.save(f"{debug_prefix}_hint_base.png")

    # --- SDXL inpaint on the hinted image (moderate strength) ---
    generator = torch.Generator(device=pipe._execution_device).manual_seed(seed)
    prompt = (f"a high-quality photo, the target region has vivid {target_color}, "
              f"rich saturated color, keep same shape and material, realistic lighting")
    negative_prompt = ("grayscale, desaturated, deformed, distorted, low quality, "
                       "extra objects, text, change of shape")

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=hint_img,             # <— use the hinted image!
        mask_image=mask_L,
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=strength,          # ~0.45–0.55 works well
        generator=generator,
    ).images[0]

    # Resize everything to result size for LAB blend
    res_w, res_h = result.size
    base_resized = base_pil.resize((res_w, res_h), Image.BICUBIC)
    mask_resized = mask_L.resize((res_w, res_h), Image.NEAREST)

    if save_debug:
        ensure_dir(f"{debug_prefix}_sdxl_raw.png")
        result.save(f"{debug_prefix}_sdxl_raw.png")

    # LAB blend: keep luminance from current base, inject chroma from SDXL; fallback if too gray
    res_np  = np.asarray(result).astype(np.float32) / 255.0
    base_np = np.asarray(base_resized).astype(np.float32) / 255.0
    lab_res  = color.rgb2lab(res_np)
    lab_base = color.rgb2lab(base_np)

    L_orig = lab_base[..., 0:1]
    ab_res = lab_res[..., 1:3]
    m = (np.asarray(mask_resized).astype(np.float32) / 255.0)[..., None]

    # Measure chroma inside mask
    chroma_mag = np.sqrt(np.sum(ab_res**2, axis=-1, keepdims=True))
    inside = m > 0.5
    mean_chroma = float(np.mean(chroma_mag[inside])) if np.any(inside) else 0.0

    # Deterministic fallback if gray
    if mean_chroma < 5.0:
        tgt = np.array(rgb, dtype=np.float32) / 255.0
        tgt_lab = color.rgb2lab(tgt.reshape(1,1,3)).reshape(1,1,3)
        ab_tgt  = tgt_lab[..., 1:3]
        alpha = np.clip(0.8, 0.0, 1.0)  # you can make this a parameter
        ab_res = ab_res * (1 - alpha*m) + (ab_tgt * (alpha*m))
        print(f"[colorize] Fallback chroma applied (mean_chroma={mean_chroma:.2f})")

    ab_blend = ab_res * m
    lab_out = np.concatenate([L_orig, ab_blend], axis=-1)
    rgb_out = np.clip(color.lab2rgb(lab_out), 0, 1)
    out_img = Image.fromarray((rgb_out * 255).astype(np.uint8))

    # Keep background equal to base image (grayscale or previously colored)
    out_img = Image.composite(out_img, base_resized, Image.fromarray((m[...,0]*255).astype(np.uint8)))
    return out_img

# ------------------- MAIN -------------------
def main():
    args = parse_args()
    ensure_dir(args.final_out)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load base image (grayscale → RGB 3-channel)
    base_gray = PILImage.open(args.image_path).convert("L")
    base_rgb  = PILImage.merge("RGB", (base_gray, base_gray, base_gray))
    orig_w, orig_h = base_rgb.size

    # Load tasks
    with open(args.tasks_json, "r") as f:
        tasks = json.load(f)
    assert isinstance(tasks, list) and len(tasks) > 0, "tasks_json must be a non-empty list"

    # ---- Load Seg-Zero models once ----
    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.reasoning_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    reasoning_model.eval()
    processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left")
    seg_model = SAM2ImagePredictor.from_pretrained(args.segmentation_model_path)

    # ---- Load SDXL inpainting once ----
    pipe = AutoPipelineForInpainting.from_pretrained(
        args.sdxl_model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        variant="fp16", use_safetensors=True
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.enable_vae_slicing(); pipe.enable_attention_slicing()
    if args.disable_safety:
        pipe.safety_checker = None; pipe.requires_safety_checker = False

    # Work image accumulates edits
    current_img = base_rgb

    # Process tasks sequentially
    for idx, task in enumerate(tasks):
        text = task["text"]
        color_spec = task.get("color", "hotpink")

        print(f"\n=== Task {idx+1}/{len(tasks)}: '{text}' -> {color_spec} ===")
        mask_bool = segzero_ground(
            reasoning_model, processor, seg_model,
            current_img, text,
            erode=args.erode, dilate=args.dilate, feather=args.feather
        )
        if args.invert_mask:
            mask_bool = ~mask_bool

        # Save mask & overlay for inspection
        mask_png = os.path.join(args.out_dir, f"mask_{idx+1}.png")
        overlay_png = os.path.join(args.out_dir, f"overlay_{idx+1}.png")
        Image.fromarray((mask_bool.astype(np.uint8)*255), mode="L").save(mask_png)
        overlay_and_save(current_img, mask_bool, overlay_png, title=f"Task {idx+1} Mask")

        # Colorize this region on the current image
        debug_prefix = os.path.join(args.out_dir, f"task{idx+1}")
        colored = sdxl_colorize_one(
            pipe, current_img, mask_bool, color_spec,
            steps=args.steps, guidance=args.guidance, strength=args.strength,
            seed=args.seed, disable_safety=args.disable_safety,
            keep_original_size=args.keep_original_size,
            save_debug=args.save_debug, debug_prefix=debug_prefix
        )

        # Next iteration uses the updated image
        current_img = colored

    # Optionally resize back
    if args.keep_original_size:
        current_img = current_img.resize((orig_w, orig_h), Image.BICUBIC)

    current_img.save(args.final_out)
    print(f"\n[OK] Saved final image → {args.final_out}")

if __name__ == "__main__":
    main()
