import os
import argparse
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import re
import cv2
from PIL import Image as PILImage, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from sam2.sam2_image_predictor import SAM2ImagePredictor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model_path", type=str, default="pretrained_models/VisionReasoner-7B")
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--text", type=str, default="What can I have if I'm thirsty?")
    parser.add_argument("--image_path", type=str, default="./assets/food.webp")
    parser.add_argument("--output_path", type=str, default="./inference_scripts/test_output_multiobject.png")

    # NEW: where to save an inpainting-ready mask (white=edit, black=keep)
    parser.add_argument("--mask_output_path", type=str, default="./inference_scripts/mask.png")
    parser.add_argument("--save_npy_mask", action="store_true", help="Also save boolean mask as mask.npy")

    # Optional mask post-processing
    parser.add_argument("--erode", type=int, default=0, help="Erode radius (px) before dilate/feather; 0 to disable")
    parser.add_argument("--dilate", type=int, default=0, help="Dilate radius (px) after erode; 0 to disable")
    parser.add_argument("--feather", type=float, default=0.0, help="Gaussian blur sigma (px) for soft edges; 0.0 to disable")

    return parser.parse_args()

def extract_bbox_points_think(output_text, x_factor, y_factor):
    # Extract JSON between <answer>...</answer>
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
                pred_bboxes.append(bx)
                pred_points.append(pt)
        except Exception:
            pred_bboxes, pred_points = [], []

    # Extract think text
    think_match = re.search(r'<think>(.*?)</think>', output_text, re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else ""

    return pred_bboxes, pred_points, think_text

def tidy_mask_bool(mask_bool, erode_px=0, dilate_px=0, feather_sigma=0.0):
    """
    mask_bool: HxW boolean, True where object is present.
    Returns uint8 (0..255) single-channel mask (white=edit) with optional cleanup.
    """
    m = mask_bool.astype(np.uint8) * 255  # 0/255

    # Morph operations with OpenCV use odd kernel sizes. Convert pixel radius -> kernel.
    if erode_px > 0:
        k = 2 * erode_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.erode(m, kernel)

    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.dilate(m, kernel)

    # Feather edges for seamless diffusion blending
    if feather_sigma and feather_sigma > 0:
        # Using GaussianBlur to create soft edges (still single-channel 0..255)
        ksize = max(1, int(6 * feather_sigma) | 1)  # odd kernel approx 6*sigma
        m = cv2.GaussianBlur(m, (ksize, ksize), feather_sigma)

        # Ensure range stays 0..255
        m = np.clip(m, 0, 255).astype(np.uint8)

    return m

def main():
    args = parse_args()

    # Reasoning model (VisionReasoner / Qwen2.5-VL)
    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.reasoning_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    segmentation_model = SAM2ImagePredictor.from_pretrained(args.segmentation_model_path)
    reasoning_model.eval()

    processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left")

    print("User question:", args.text)

    QUESTION_TEMPLATE = (
        "Please find \"{Question}\" with bboxs and points."
        "Compare the difference between object(s) and find the most closely matched object(s)."
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
        "Output the bbox(es) and point(s) inside the interested object(s) in JSON format."
        "i.e., <think> thinking process here </think>"
        "<answer>{Answer}</answer>"
    )

    image = PILImage.open(args.image_path).convert("RGB")
    original_width, original_height = image.size
    resize_size = 840
    x_factor, y_factor = original_width / resize_size, original_height / resize_size

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image.resize((resize_size, resize_size), PILImage.BILINEAR)},
            {
                "type": "text",
                "text": QUESTION_TEMPLATE.format(
                    Question=args.text.lower().strip("."),
                    Answer=(
                        "[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}, "
                        "{\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410]}]"
                    )
                )
            }
        ]
    }]

    # Prep inputs
    text = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
    image_inputs, video_inputs = process_vision_info([messages])
    inputs = processor(
        text=text, images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to("cuda")

    # Generate
    generated_ids = reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(output_text[0])

    bboxes, points, think = extract_bbox_points_think(output_text[0], x_factor, y_factor)
    print("Points:", points, "Count:", len(points))
    print("Thinking process:", think)

    # Segmentation
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        mask_all = np.zeros((image.height, image.width), dtype=bool)
        segmentation_model.set_image(image)
        if len(points) == 0 or len(bboxes) == 0:
            print("[WARN] No objects parsed from reasoning output. Exiting early.")
        else:
            for bbox, point in zip(bboxes, points):
                masks, scores, _ = segmentation_model.predict(
                    point_coords=[point],
                    point_labels=[1],
                    box=bbox
                )
                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                mask = masks[0].astype(bool)
                mask_all = np.logical_or(mask_all, mask)

    # --- NEW: Save inpainting-ready mask ---
    os.makedirs(os.path.dirname(args.mask_output_path), exist_ok=True)
    mask_u8 = tidy_mask_bool(
        mask_all,
        erode_px=max(0, args.erode),
        dilate_px=max(0, args.dilate),
        feather_sigma=max(0.0, args.feather),
    )
    PILImage.fromarray(mask_u8, mode="L").save(args.mask_output_path)
    print(f"[OK] Saved mask for inpainting → {args.mask_output_path}")

    if args.save_npy_mask:
        np.save(os.path.join(os.path.dirname(args.mask_output_path), "mask.npy"), mask_all)
        print(f"[OK] Saved boolean mask → {os.path.join(os.path.dirname(args.mask_output_path), 'mask.npy')}")

    # --- Keep your visualization (overlay) ---
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image, alpha=0.6)
    # For overlay, convert boolean to RGB
    mask_rgb = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    mask_rgb[mask_all] = [255, 0, 0]
    plt.imshow(mask_rgb, alpha=0.4)
    plt.title('Image with Predicted Mask')
    plt.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    plt.savefig(args.output_path, dpi=150)
    plt.close()
    print(f"[OK] Saved overlay → {args.output_path}")

if __name__ == "__main__":
    main()
