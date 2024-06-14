import cv2
import os
from pathlib import Path

path = Path().resolve()
image_folder = str(path/"pics/spatio-temporal")
video_folder = str(path/"videos")

frame = cv2.imread(os.path.join(image_folder, "mvtarg_img1.png"))
h, w, l = frame.shape
print("Dimensions: ", h, w, l)

video_name = os.path.join(video_folder, "video_fast.avi")
fps = 10
video = cv2.VideoWriter(video_name, 0, fps, (w,h))

for i in range(1, 51):
    video.write(cv2.imread(os.path.join(image_folder, f"mvtarg_img{i}.png")))

cv2.destroyAllWindows()
video.release()