import cv2
import os
import numpy as np
from tqdm import tqdm
from silhoutte_pre.silhoutte.scripts.deeplab_sil import get_silhouette


def process_video(video_path, output_video="output_silhouette_video.mp4", 
                  output_folder="output_frames", box_size=(512, 512)):

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    out = cv2.VideoWriter(output_video, 
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps,box_size)

    print("Processing video...")

    for frame_number in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Skipping frame {frame_number}")
            continue

        # Extract silhouette
        silhouette = get_silhouette(frame, box_size=box_size)

        if silhouette is not None:
            sil_path = os.path.join(output_folder, f"silhouette_{frame_number:04d}.png")
            cv2.imwrite(sil_path, silhouette)
            out.write(silhouette)

        if len(silhouette.shape) == 2:
            silhouette = cv2.cvtColor(silhouette, cv2.COLOR_GRAY2BGR)
            # Add frame to silhouette video
        silhouette = cv2.resize(silhouette, box_size)
        sil_path = os.path.join(output_folder, f"silhouette_{frame_number:04d}.png")
        cv2.imwrite(sil_path, silhouette)
        out.write(silhouette)

    cap.release()
    out.release()
    print(f"Silhouette frames saved in: {output_folder}")
    print(f"Silhouette video saved as: {output_video}")

    return output_folder



