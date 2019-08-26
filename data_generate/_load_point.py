import numpy as np
import cv2
import os

input_folder = '1a6f615e8b1b5ae4dbbc9440457e303e/temp'
for index in range(24):
    point = np.load(os.path.join(input_folder,'{0:02d}.npy'.format(int(index))))
    print(len(point))
    img = cv2.imread(os.path.join(input_folder,'{0:02d}.png'.format(int(index))))
    X,Y,Z = point.T
    F = 284
    h = (-Y)/(-Z)*F + 256/2.0
    w = X/(-Z)*F + 256/2.0
    h = np.minimum(np.maximum(h, 0), 255)
    w = np.minimum(np.maximum(w, 0), 255)
    img[np.round(h).astype(int), np.round(w).astype(int), 2] = 0
    img[np.round(h).astype(int), np.round(w).astype(int), 1] = 255
    cv2.imwrite(os.path.join(input_folder,'{0:02d}.png'.format(int(index))).replace('.png','_prj.png'), img)

