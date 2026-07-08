import json
import cv2
import numpy as np
import easyocr
from jiwer import wer
from Levenshtein import distance as levenshtein_distance
import pandas as pd
import os

LABEL_JSON = "D:/FYP CODE/EASYOCR LATEST TEST/Input_Images_And_Annotations"  #annotations json file
IMAGE_FOLDER = "D:/FYP CODE/EASYOCR LATEST/Floor Plan PNG"  #input image folder
OUTPUT_CSV = "D:/FYP CODE/EASYOCR LATEST/easyocr_recognition_RAW.csv"  #result csv
VIS_FOLDER = "D:/FYP CODE/EASYOCR LATEST/annotated_image_RAW" #output image folder
LANGUAGES = ["en"]
GPU = True

# --- UTILITY FUNCTIONS (Metrics & Helpers) ---

def cer(gt, pred):
    gt = "" if gt is None else str(gt)
    pred = "" if pred is None else str(pred)
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return levenshtein_distance(gt, pred) / len(gt)

def percent_bbox_to_px(bbox, img_w, img_h):
    x_pct = float(bbox.get("x", 0))
    y_pct = float(bbox.get("y", 0))
    w_pct = float(bbox.get("width", 0))
    h_pct = float(bbox.get("height", 0))
    x1 = int(round(x_pct / 100.0 * img_w))
    y1 = int(round(y_pct / 100.0 * img_h))
    x2 = int(round((x_pct + w_pct) / 100.0 * img_w))
    y2 = int(round((y_pct + h_pct) / 100.0 * img_h))
    # Clip to image bounds
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))
    return [x1, y1, x2, y2]

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(1, (boxA[2]-boxA[0])) * max(1, (boxA[3]-boxA[1]))
    areaB = max(1, (boxB[2]-boxB[0])) * max(1, (boxB[3]-boxB[1]))
    union = areaA + areaB - inter
    return inter/union if union > 0 else 0.0

def find_image_file(json_filename, image_folder):
    if not json_filename: return None
    path = os.path.join(image_folder, json_filename)
    if os.path.isfile(path): return path
    if '-' in json_filename:
        clean_name = json_filename.split('-', 1)[1]
        path = os.path.join(image_folder, clean_name)
        if os.path.isfile(path): return path
        base_name = os.path.splitext(clean_name)[0]
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            path = os.path.join(image_folder, base_name + ext)
            if os.path.isfile(path): return path
    base_name = os.path.splitext(json_filename)[0]
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        path = os.path.join(image_folder, base_name + ext)
        if os.path.isfile(path): return path
    return None

def draw_text_safe(img, text, x, y, color, scale=0.9): 
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    (t_w, t_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    
    if y - t_h - 5 < 0:
        y = t_h + 5

    cv2.rectangle(img, (x, y - t_h - 5), (x + t_w, y + baseline - 5), color, -1)
    cv2.putText(img, text, (x, y - 5), font, scale, (255, 255, 255), thickness)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    os.makedirs(VIS_FOLDER, exist_ok=True)

    with open(LABEL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data if isinstance(data, list) else [data]
    reader = easyocr.Reader(LANGUAGES, gpu=GPU)
    all_rows = []

    print(f"Found {len(items)} items in JSON.")

    for item in items:
        # --- 1. Find Image ---
        raw_filename = item.get("file_upload")
        if raw_filename: raw_filename = os.path.basename(raw_filename)
        
        image_path = find_image_file(raw_filename, IMAGE_FOLDER)
        if not image_path:
            print(f"[WARNING] Image not found for ID {item.get('id')}: {raw_filename}")
            continue
        
        print(f"\nProcessing ID {item.get('id')}: {os.path.basename(image_path)}")

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print("cv2 failed to load image. Skipping.")
            continue
        
        orig_h, orig_w = img_bgr.shape[:2]

        # --- 2. Run OCR (NO PREPROCESSING) ---
        # Pass the raw BGR image directly to EasyOCR
        preds = reader.readtext(img_bgr, detail=1)
        
        pred_items = []
        for poly, text, conf in preds:
            xs = [int(round(p[0])) for p in poly]
            ys = [int(round(p[1])) for p in poly]
            pred_items.append({
                "box": [min(xs), min(ys), max(xs), max(ys)], 
                "text": text.strip(), 
                "conf": conf
            })

        # --- 3. Parse Annotations ---
        ann = item.get("annotations") or item.get("predictions") or []
        results = ann[0].get("result", []) if ann else []
        rects = {r["id"]: r for r in results if r.get("type") == "rectangle"}
        texts = [r for r in results if r.get("type") == "textarea"]

        # Create visualization image (copy of original, no grayscale conversion needed)
        vis_img = img_bgr.copy()

        # A. Draw All Predictions (RED)
        for p in pred_items:
            x1, y1, x2, y2 = p["box"]
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2) 
            draw_text_safe(vis_img, f"PRED: {p['text']}", x1, y1, (0, 0, 255))

        # B. Match with Ground Truth
        for t in texts:
            val = t.get("value", {})
            gt_text = str(val["text"][0]) if isinstance(val.get("text"), list) else str(val.get("text",""))
            rid = t.get("id")
            
            if rid not in rects: continue

            bbox_px = percent_bbox_to_px(rects[rid]["value"], orig_w, orig_h)
            
            # Draw Ground Truth (GREEN)
            gx1, gy1, gx2, gy2 = bbox_px
            cv2.rectangle(vis_img, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
            draw_text_safe(vis_img, f"GT: {gt_text}", gx1, gy1 - 25, (0, 255, 0))

            # IoU Matching
            best_j = None
            best_iou = 0.0
            for j, p in enumerate(pred_items):
                candidate_iou = iou(bbox_px, p["box"])
                if candidate_iou > best_iou:
                    best_iou = candidate_iou
                    best_j = j

            matched_text = ""
            matched_conf = ""
            IOU_THRESH = 0.3 
            
            if best_j is not None and best_iou >= IOU_THRESH:
                matched_text = pred_items[best_j]["text"]
                matched_conf = pred_items[best_j]["conf"]

            c = cer(gt_text, matched_text)
            w = wer(gt_text, matched_text) if len(gt_text.split())>0 else (0.0 if matched_text=="" else 1.0)

            all_rows.append({
                "image_filename": os.path.basename(image_path),
                "region_id": rid,
                "gt_text": gt_text,
                "pred_text": matched_text,
                "cer": c,
                "wer": w,
                "iou_with_best": best_iou,
                "matched_conf": matched_conf
            })
            
            print(f"GT: {gt_text} | Pred: {matched_text}")

        # Save visualization
        vis_filename = f"vis_{os.path.basename(image_path)}"
        vis_path = os.path.join(VIS_FOLDER, vis_filename)
        cv2.imwrite(vis_path, vis_img)
        print(f"Saved visualization to: {vis_path}")

    # --- 4. Save Report ---
    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No evaluated regions found.")
    else:
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"\nSaved RAW (no preprocess) report to: {OUTPUT_CSV}")
        print(f"Total Regions: {len(df)}")
        print(f"Avg CER: {df['cer'].mean():.4f}")
        print(f"Avg WER: {df['wer'].mean():.4f}")