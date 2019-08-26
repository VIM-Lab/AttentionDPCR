import numpy as np
import os

image_prefix = '/media/tree/backup/projects/AttentionBased/data/image'
point_prefix = '/media/tree/backup/projects/AttentionBased/data/pointcloud_16384'
output_folder = '/media/tree/backup/projects/AttentionBased/data/data'

image_path = []
point_path = []
# common_path = []

image = os.listdir(image_prefix)
image_path = [os.path.join(image_prefix, i) for i in image]
point = os.listdir(point_prefix)
point_path = [os.path.join(point_prefix, i) for i in point]


def work(idx):
    # batch_img = [ np.load(self.image_path[idx+i]) for i in range(self.batch_size) ]
    # batch_label = [ np.load(self.point_path[idx+i]) for i in range(self.batch_size) ]
    # batch_model_id = []

    batch_img = []
    batch_label = []

    for i in range(self.batch_size):

        image_path = image_path[idx+i]
        point_path = point_path[idx+i]
        # single_model_id = image_path.split('/')[-1]
        # image = cv2.imread(image_path)
        image_array = np.load(image_path)
        point_array = np.load(point_path)

        batch_img.append(image_array)
        batch_label.append(point_array)

        # batch_model_id.append(single_model_id)

    return np.array(batch_img), np.array(batch_label)
    # return np.array(batch_img), np.array(batch_label), batch_model_id

def fetch():
    idx = 0
    while idx <= len(image_path):
        work(idx)
        idx += 32

if __name__ == '__main__':
    import time
    load_start = time.time()
    fetch()
    load_end = time.time()
    print('load_data', load_end - load_start)
