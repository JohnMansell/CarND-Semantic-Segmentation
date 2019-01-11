import cv2
import os

import_folder = 'runs/1547221949.9810524'
video_name = 'video_8.avi'

images = [img for img in os.listdir(import_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(import_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 6, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(import_folder, image)))

cv2.destroyAllWindows()
video.release