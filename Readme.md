# Cricket Ball Tracker

EdgeFleet AI/ML Assessment repository.

## Structure
- `code/` - inference, training, tracker and utils
- `annotations/` - CSV annotation files
- `results/` - processed videos / overlays
- `report.md` - analysis & results (convert to PDF as needed)

## Quick start (no venv required)
1. Install dependencies:
`pip install -r requirements.txt`

2. Run inference ( YOLOv8x; requires internet to auto-download):
`python code/inference.py --input_video path/to/video.mov --model_path yolov8x.pt --output_video results/processed_video.mp4 --output_csv results/annotations.csv --use_model`
