# Report — Cricket Ball Tracker

## 1. Objective
Detect and track cricket ball centroid per frame from a single static camera.

## 2. Approach
- YOLO-based detector (ultralytics YOLOv8) when model available.
- Kalman filter tracker for smoothing/prediction and handling missed detections.
- HSV+motion fallback for quick runs when a trained detector is missing or fails.

## 3. Pipeline
1. Run detector on each frame (or detect in high-res tile around predicted location).
2. If no detection, run HSV+motion fallback.
3. Update the Kalman tracker.
4. Save per-frame `frame,x,y,visible` CSV and processed overlay video.

## 4. Experiments & tuning
- Lower detection confidence for small objects (e.g., 0.15–0.25).
- Use larger inference image sizes (imgsz 896+) or tile+upsample technique.
- Train with motion-blur augmentations for better robustness.

## 5. Results
Include screenshots and quantitative metrics (precision/recall) here after you run the pipeline.

## 6. Failure cases & improvements
- Motion blur / tiny objects: use higher shutter/ FPS or train with blur.
- Occlusion: multi-camera or re-id may be required.

## 7. Reproducibility
Commands used, dataset splits, training commands, hyperparameters (put them here).

