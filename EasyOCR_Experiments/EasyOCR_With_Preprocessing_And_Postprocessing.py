import json
import cv2
import numpy as np
import easyocr
from jiwer import wer
from Levenshtein import distance as levenshtein_distance
import pandas as pd
import os
import re

LABEL_JSON = "D:/FYP CODE/EasyOCR_Experiments/Input_Images_And_Annotations/FloorPlanTextAnnotations.json"  #annotations json file
IMAGE_FOLDER = "D:/FYP CODE/EasyOCR_Experiments/Input_Images_And_Annotations/Self Collected Floor Plan Input Image"  #input image folder
OUTPUT_CSV = "D:/FYP CODE/EasyOCR_Experiments/Results/EasyOCR_Results_WithPreprocessingAndPostprocessing.csv"  #result csv
VIS_FOLDER = "D:/FYP CODE/EasyOCR_Experiments/EasyOCR_Output_WithPreprocessingAndPostprocessing" #output image folder
LANGUAGES = ["en"]
GPU = True

def post_process_dimensions(text):
    if not text: return ""
    # Fix common OCR swaps: Q->0, I/l/[ -> 1, Z -> 2
    text = re.sub(r'(?<=\d)Q|Q(?=\d)', '0', text)
    text = re.sub(r'(?<=\d)[Il\[]|[Il\[](?=\d)', '1', text)
    text = re.sub(r'(?<=\d)Z|Z(?=\d)', '2', text)
    
    # Standardize multiplication sign (x, *, %, X)
    text = re.sub(r'\s*[\*xX%]\s*', ' x ', text)
    
    # Fix double single-quotes to double-quote
    text = text.replace("''", '"')
    
    # Standardize dash spacing in dimensions (e.g., 10'-0" X 5'-0")
    text = re.sub(r"(\d+')\s*-\s*(\d+\")", r"\1-\2", text)
    return text.strip()

# Fuzzy match room names to a known dictionary
def fuzzy_correct_room(text):
    ROOM_DICT = ["KITCHEN", "BATH", "MASTER BEDROOM", "M. BEDROOM", "W.I.C.", 
                 "GARAGE", "DINING", "FOYER", "LIVING", "MUDROOM", "ENTRY", "NOOK"]
    text_upper = text.upper().strip()
    if not text_upper: return text
    
    for room in ROOM_DICT:
        # If the prediction is very close (within 2 chars), correct it
        if levenshtein_distance(text_upper, room) <= 2:
            return room
    return text

def sauvola_threshold(gray, window_size=25, k=0.25, R=128):
    if window_size % 2 == 0:
        window_size += 1
    img = gray.astype(np.float32)
    mean = cv2.boxFilter(img, ddepth=-1, ksize=(window_size, window_size), normalize=True)
    mean_sq = cv2.boxFilter(img * img, ddepth=-1, ksize=(window_size, window_size), normalize=True)
    var = mean_sq - mean * mean
    std = np.sqrt(np.clip(var, 0, None))
    
    thresh = mean * (1 + k * (std / R - 1))
    binary = (img > thresh).astype(np.uint8) * 255
    return binary

def morphological(binary):
    k1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k1, iterations=1)
    return closing

def run_preprocessing_pipeline(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sauvola = sauvola_threshold(gray)
    sauvola_inv = cv2.bitwise_not(sauvola)
    processed_inv = morphological(sauvola_inv)
    processed = cv2.bitwise_not(processed_inv)
    return processed

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
    return inter/union if union>0 else 0.0

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

# Simple normalization for calculating baseline/RAW scores
def basic_normalize(s):
    if not s: return ""
    return str(s).lower().strip()

# Advanced normalization utilizing our custom post-processing rules
def normalize_text(s):
    if not s: return ""
    # Convert to uppercase for consistent matching
    s = str(s).upper().strip()
    # Apply post-processing
    s = post_process_dimensions(s)
    s = fuzzy_correct_room(s)
    # Return lower for the metric comparison
    return s.lower()

def build_pred_items(preds):
    items = []
    for poly, text, conf in preds:
        xs = [int(round(p[0])) for p in poly]
        ys = [int(round(p[1])) for p in poly]
        items.append({
            "box": [min(xs), min(ys), max(xs), max(ys)],
            "text": text.strip(),
            "conf": conf
        })
    return items

def find_best_match(gt_box, pred_items):
    best = None
    best_iou = 0.0
    for p in pred_items:
        iou_val = iou(gt_box, p["box"])
        if iou_val > best_iou:
            best_iou = iou_val
            best = p
    return best, best_iou

def score(item, iou_val):
    return 0.7 * item["conf"] + 0.3 * iou_val

if __name__ == "__main__":
    os.makedirs(VIS_FOLDER, exist_ok=True)

    with open(LABEL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data if isinstance(data, list) else [data]
    reader = easyocr.Reader(LANGUAGES, gpu=GPU)
    all_rows = []

    print(f"Found {len(items)} items in JSON.")

    for item in items:
        raw_filename = item.get("file_upload")
        if raw_filename: raw_filename = os.path.basename(raw_filename)
        
        image_path = find_image_file(raw_filename, IMAGE_FOLDER)
        if not image_path: continue
        
        print(f"\nProcessing ID {item.get('id')}: {os.path.basename(image_path)}")

        img_bgr = cv2.imread(image_path)
        if img_bgr is None: continue
        
        orig_h, orig_w = img_bgr.shape[:2]

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        processed_full = run_preprocessing_pipeline(img_bgr)

        # 1. GENERATE HORIZONTAL
        preds_raw_h = reader.readtext(gray, detail=1, x_ths=1.5)
        preds_proc_h = reader.readtext(processed_full, detail=1, x_ths=1.5)

        # 2. GENERATE VERTICAL
        preds_raw_v = reader.readtext(gray, detail=1, x_ths=1.5, rotation_info=[90, 270])
        preds_proc_v = reader.readtext(processed_full, detail=1, x_ths=1.5, rotation_info=[90, 270])

        pred_items_raw_h = build_pred_items(preds_raw_h)
        pred_items_proc_h = build_pred_items(preds_proc_h)
        pred_items_raw_v = build_pred_items(preds_raw_v)
        pred_items_proc_v = build_pred_items(preds_proc_v)

        vis_img = cv2.cvtColor(processed_full, cv2.COLOR_GRAY2BGR)

        ann = item.get("annotations") or item.get("predictions") or []
        results = ann[0].get("result", []) if ann else []
        rects = {r["id"]: r for r in results if r.get("type") == "rectangle"}
        texts = [r for r in results if r.get("type") == "textarea"]

        for t in texts:
            val = t.get("value", {})
            gt_text = str(val["text"][0]) if isinstance(val.get("text"), list) else str(val.get("text",""))
            rid = t.get("id")
            
            if rid not in rects: continue

            bbox_px = percent_bbox_to_px(rects[rid]["value"], orig_w, orig_h)
            gx1, gy1, gx2, gy2 = bbox_px
            
            cv2.rectangle(vis_img, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
            draw_text_safe(vis_img, f"GT: {gt_text}", gx1, gy1 - 25, (0, 255, 0))

            box_w = gx2 - gx1
            box_h = gy2 - gy1

            if box_h > box_w:
                pool_raw = pred_items_raw_v
                pool_proc = pred_items_proc_v
                source_tag = "_vert"
            else:
                pool_raw = pred_items_raw_h
                pool_proc = pred_items_proc_h
                source_tag = "_horiz"

            best_raw, iou_raw = find_best_match(bbox_px, pool_raw)
            best_proc, iou_proc = find_best_match(bbox_px, pool_proc)

            IOU_THRESH = 0.3
            candidates = []

            if best_raw is not None and iou_raw >= IOU_THRESH:
                candidates.append(("raw", best_raw, iou_raw))

            if best_proc is not None and iou_proc >= IOU_THRESH:
                candidates.append(("processed", best_proc, iou_proc))

            raw_pred = ""
            processed_pred = ""
            matched_conf = ""
            best_iou = 0.0
            source = "none"

            if len(candidates) > 0:
                candidates.sort(key=lambda x: score(x[1], x[2]), reverse=True)
                source, best_item, best_iou = candidates[0]
                
                # 1. Capture Raw Prediction
                raw_pred = best_item["text"]
                
                # 2. Apply Post Processing
                processed_pred = post_process_dimensions(raw_pred)
                processed_pred = fuzzy_correct_room(processed_pred)
                
                matched_conf = best_item["conf"]
                source += source_tag

            # --- CALCULATE METRICS ---
            # Baseline (RAW) metrics
            gt_raw_norm = basic_normalize(gt_text)
            pred_raw_norm = basic_normalize(raw_pred)
            c_raw = cer(gt_raw_norm, pred_raw_norm)
            w_raw = wer(gt_raw_norm, pred_raw_norm) if len(gt_raw_norm.split())>0 else (0.0 if pred_raw_norm=="" else 1.0)

            # Post-processed metrics
            gt_post_norm = normalize_text(gt_text)
            pred_post_norm = normalize_text(processed_pred)
            c_post = cer(gt_post_norm, pred_post_norm)
            w_post = wer(gt_post_norm, pred_post_norm) if len(gt_post_norm.split())>0 else (0.0 if pred_post_norm=="" else 1.0)

            all_rows.append({
                "image_filename": os.path.basename(image_path),
                "region_id": rid,
                "gt_text": gt_text,
                "raw_pred_text": raw_pred,           # New Column
                "post_processed_text": processed_pred, # New Column
                "cer_raw": c_raw,                    # New Column
                "wer_raw": w_raw,                    # New Column
                "cer_post": c_post,                  # New Column
                "wer_post": w_post,                  # New Column
                "iou_with_best": best_iou,
                "matched_conf": matched_conf,
                "source": source
            })
            
            print(f"GT: {gt_text} | Raw Pred: {raw_pred} | Clean Pred: {processed_pred}")

            if processed_pred:
                draw_text_safe(vis_img, f"PR: {processed_pred}", gx1, gy2 + 20, (0, 0, 255))

        vis_filename = f"vis_{os.path.basename(image_path)}"
        vis_path = os.path.join(VIS_FOLDER, vis_filename)
        cv2.imwrite(vis_path, vis_img)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No evaluated regions found.")
    else:
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"\nSaved report to: {OUTPUT_CSV}")
        print(f"\n--- RAW RESULTS (Before Cleaning) ---")
        print(f"Avg CER: {df['cer_raw'].mean():.4f}")
        print(f"Avg WER: {df['wer_raw'].mean():.4f}")
        print(f"\n--- POST-PROCESSED RESULTS (After Cleaning) ---")
        print(f"Avg CER: {df['cer_post'].mean():.4f}")
        print(f"Avg WER: {df['wer_post'].mean():.4f}")