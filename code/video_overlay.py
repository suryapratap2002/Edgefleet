import cv2
import pandas as pd

def overlay_from_csv(video_in, csv_path, video_out):
    df = pd.read_csv(csv_path)
    cap = cv2.VideoCapture(video_in)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_id < len(df):
            row = df.iloc[frame_id]
            if int(row['visible']) == 1:
                x = int(row['x']); y = int(row['y'])
                cv2.circle(frame, (x,y), 6, (0,0,255), -1)
        writer.write(frame)
        frame_id += 1
    cap.release(); writer.release()

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--video_in', required=True)
    p.add_argument('--csv', required=True)
    p.add_argument('--video_out', required=True)
    args = p.parse_args()
    overlay_from_csv(args.video_in, args.csv, args.video_out)
