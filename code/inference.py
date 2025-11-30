"""
Robust inference pipeline for cricket ball detection & tracking.

Usage examples (run from repo root):
# try using a YOLO model (ultralytics will auto-download yolov8n.pt if you pass that name and have internet)
python code/inference.py --input_video path/to/video.mov --model_path yolov8n.pt --output_video results/processed_video.mp4 --output_csv results/annotations.csv --use_model

# force fallback (HSV+motion)
python code/inference.py --input_video path/to/video.mov --model_path none --output_video results/processed_video.mp4 --output_csv results/annotations.csv
"""
import argparse
import os
import cv2
import numpy as np
from tracker import KalmanBallTracker
from utils import ball_hsv_mask, largest_contour_centroid

# Try to import YOLO only if available
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


def detect_ball_yolo(model, frame, conf_thresh=0.25):
    """Run YOLO model on a single frame, return cx,cy,visible"""
    res = model(frame)[0]
    if len(res.boxes) == 0:
        return -1, -1, False
    best = None
    best_conf = 0.0
    for box in res.boxes:
        conf = float(box.conf[0])
        if conf >= conf_thresh and conf > best_conf:
            best_conf = conf
            x1, y1, x2, y2 = box.xyxy[0]
            best = (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))
    if best is None:
        return -1, -1, False
    return best[0], best[1], True


def detect_ball_hsv_motion(frame, prev_gray=None, lower=(5, 90, 80), upper=(30, 255, 255)):
    """HSV + optional motion fallback detector (fast, simple)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_np = np.array(lower)
    upper_np = np.array(upper)
    mask = cv2.inRange(hsv, lower_np, upper_np)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    if prev_gray is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        _, dmask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        dmask = cv2.medianBlur(dmask, 5)
        mask = cv2.bitwise_and(mask, dmask)

    # find largest contour
    centroid = largest_contour_centroid(mask, min_area=10)
    if centroid is None:
        return -1, -1, False
    cx, cy = centroid
    return float(cx), float(cy), True


def tile_and_detect(model, frame, center, tile_size=200, upsample_size=896, conf_thresh=0.2):
    """
    Optional helper: crop a tile around `center` (x,y), upsample it, run detector,
    and return detection mapped back to full-frame coordinates.
    Use to improve small-object detection near predicted position.
    """
    h, w = frame.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    half = tile_size // 2
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(w, cx + half)
    y1 = min(h, cy + half)
    tile = frame[y0:y1, x0:x1].copy()
    if tile.size == 0:
        return -1, -1, False
    tile_resized = cv2.resize(tile, (upsample_size, upsample_size), interpolation=cv2.INTER_CUBIC)
    res = model(tile_resized)[0]
    if len(res.boxes) == 0:
        return -1, -1, False
    best = None
    best_conf = 0.0
    for box in res.boxes:
        conf = float(box.conf[0])
        if conf >= conf_thresh and conf > best_conf:
            best_conf = conf
            x1b, y1b, x2b, y2b = box.xyxy[0]
            bx = float((x1b + x2b) / 2.0)
            by = float((y1b + y2b) / 2.0)
            best = (bx, by)
    if best is None:
        return -1, -1, False
    # map back to original coordinates
    scale_x = (x1 - x0) / upsample_size
    scale_y = (y1 - y0) / upsample_size
    mapped_x = x0 + best[0] * scale_x
    mapped_y = y0 + best[1] * scale_y
    return mapped_x, mapped_y, True


def run(video_path, model_path, out_video, out_csv, use_model=True, conf_thresh=0.25, hsv_lower=(5,90,80), hsv_upper=(30,255,255)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    model = None
    have_model = False
    if use_model and model_path and model_path.lower() != 'none':
        if os.path.exists(model_path):
            if YOLO_AVAILABLE:
                try:
                    model = YOLO(model_path)
                    have_model = True
                    print('Loaded YOLO model from:', model_path)
                except Exception as e:
                    print('Failed to load YOLO model:', e)
                    have_model = False
            else:
                print('ultralytics not installed; falling back to HSV detector')
                have_model = False
        else:
            if YOLO_AVAILABLE:
                try:
                    model = YOLO(model_path)  # may auto-download if recognized name
                    have_model = True
                    print('Downloaded/loaded YOLO model:', model_path)
                except Exception as e:
                    print('Could not load requested YOLO model:', e)
                    have_model = False
            else:
                print(f"Model file {model_path} not found and ultralytics not available. Using HSV fallback.")
                have_model = False
    else:
        print('Model disabled or not provided. Using HSV+motion fallback.')

    tracker = KalmanBallTracker()
    prev_gray = None
    csv_lines = ['frame,x,y,visible']
    traj = []
    frame_id = 0

    # parameters for tiled detection around tracker prediction
    enable_tile = True
    tile_size = 220
    upsample_size = 896
    tile_conf_thresh = 0.18

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cx, cy, vis = -1, -1, False

        # try model full-frame first if available
        if have_model and model is not None:
            try:
                cx, cy, vis = detect_ball_yolo(model, frame, conf_thresh)
                # if model didn't detect but we have a tracker prediction, try tile detection
                if not vis and tracker.initialized and enable_tile:
                    pred_x, pred_y = tracker.predict()
                    tx, ty, tv = tile_and_detect(model, frame, (pred_x, pred_y), tile_size=tile_size, upsample_size=upsample_size, conf_thresh=tile_conf_thresh)
                    if tv:
                        cx, cy, vis = tx, ty, True
            except Exception as e:
                print('YOLO error on frame', frame_id, e)
                vis = False

        # fallback to HSV+motion if model not used or not detecting
        if not vis:
            cx, cy, vis = detect_ball_hsv_motion(frame, prev_gray, lower=hsv_lower, upper=hsv_upper)

        if vis:
            tracker.update(cx, cy)
            csv_lines.append(f"{frame_id},{cx:.2f},{cy:.2f},1")
        else:
            px, py = tracker.predict()
            cx, cy = px, py
            csv_lines.append(f"{frame_id},-1,-1,0")

        traj.append((cx, cy))
        # draw small trajectory (recent)
        max_traj = 200
        for i in range(max(0, len(traj)-max_traj), len(traj)):
            if i == 0: continue
            p1 = traj[i-1]; p2 = traj[i]
            try:
                cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0,255,0), 2)
            except Exception:
                pass
        # draw current centroid
        try:
            cv2.circle(frame, (int(cx), int(cy)), 5, (0,0,255), -1)
        except Exception:
            pass

        writer.write(frame)
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_id += 1

    cap.release(); writer.release()
    with open(out_csv, 'w') as f:
        f.write('\n'.join(csv_lines))
    print('Finished. Output:', out_video, out_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', required=True)
    parser.add_argument('--model_path', required=False, default='yolov8n.pt')
    parser.add_argument('--output_video', required=True)
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--use_model', action='store_true')
    parser.add_argument('--conf_thresh', type=float, default=0.25)
    parser.add_argument('--hsv_lower', type=str, default='5,90,80', help='comma separated H,S,V lower')
    parser.add_argument('--hsv_upper', type=str, default='30,255,255', help='comma separated H,S,V upper')
    args = parser.parse_args()

    hsv_lower = tuple(int(x) for x in args.hsv_lower.split(','))
    hsv_upper = tuple(int(x) for x in args.hsv_upper.split(','))
    use_model_flag = args.use_model or (args.model_path and args.model_path.lower() != 'none')

    run(args.input_video, args.model_path, args.output_video, args.output_csv, use_model=use_model_flag, conf_thresh=args.conf_thresh, hsv_lower=hsv_lower, hsv_upper=hsv_upper)
