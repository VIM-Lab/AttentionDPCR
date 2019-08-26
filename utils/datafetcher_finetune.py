import os
import numpy as np
import h5py
import threading
import Queue
import math

shapenet_category_to_id = {
'airplane'	: '02691156',
'bench'		: '02828884',
'cabinet'	: '02933112',
'car'		: '02958343',
'chair'		: '03001627',
'lamp'		: '03636649',
'monitor'	: '03211117',
'rifle'		: '04090263',
'sofa'		: '04256520',
'speaker'	: '03691459',
'table'		: '04379243',
'telephone'	: '04401088',
'vessel'	: '04530566'
}

#image_path -> input point path
#point_path -> gt point path

class DataFetcher(threading.Thread):
    def __init__(self, mode, batch_size = 32, epoch = 10):
        super(DataFetcher, self).__init__()
        self.stopped = False
        self.epoch = epoch
        self.current_epoch = 0
        self.queue = Queue.Queue(2)
        self.batch_size = batch_size
        self.mode = mode
        self.hierarchies = [1024,4096,16384]
        self.point_path_densepcr = {}
        if self.mode == 'train':
            # self.image_path = '/media/tree/backup/projects/AttentionBased/data/train/image_192_256_12'
            self.image_path = '/media/tree/backup/projects/AttentionBased/data/train/image_256_256_12'
            self.point_path_densepcr[1024] = '/media/tree/backup/projects/AttentionBased/data/train/point_1024_12'
            self.point_path_densepcr[16384] = '/media/tree/backup/projects/AttentionBased/data/train/point_16384_12'
            # self.point_path = '/media/tree/backup/projects/AttentionBased/data/train/point_1024_12'
        else:
            # self.image_path = '/media/tree/backup/projects/AttentionBased/data/test/image_192_256_12'
            self.image_path = '/media/tree/backup/projects/AttentionBased/data/test/image_256_256_12'
            self.point_path_densepcr[1024] = '/media/tree/backup/projects/AttentionBased/data/test/point_1024_12'
            self.point_path_densepcr[16384] = '/media/tree/backup/projects/AttentionBased/data/test/point_16384_12'
            # self.point_path = '/media/tree/backup/projects/AttentionBased/data/test/point_1024_12'
        self.iter, self.cats_batches = self.calculate_cat_batch_number()
    
    def calculate_cat_batch_number(self):
        count = 0
        cats = shapenet_category_to_id.values()
        cat_batch_number = []
        for cat in cats:
            with h5py.File(os.path.join(self.image_path, '{}.h5'.format(cat)), 'r') as f:
                batch_number = f['image'].shape[0] / self.batch_size
                cat_batch_number.append(batch_number)
                count += batch_number
        cats_batches = dict(zip(cats, cat_batch_number))
        print(cats_batches)
        return count, cats_batches

    def run(self):
        if self.mode == 'train':
            while self.current_epoch < self.epoch:
                for cat, batch in self.cats_batches.iteritems():
                    with h5py.File(os.path.join(self.image_path, '{}.h5'.format(cat)), 'r') as fi:
                        with h5py.File(os.path.join(self.point_path_densepcr[1024], '{}.h5'.format(cat)), 'r') as fp1:
                            with h5py.File(os.path.join(self.point_path_densepcr[16384], '{}.h5'.format(cat)), 'r') as fp3:
                                for i in range(0, batch * self.batch_size, self.batch_size):
                                    if self.stopped:
                                        break
                                    self.queue.put((fi['image'][i:i+self.batch_size].astype('float32') / 255.0, fp1['point'][i:i+self.batch_size], fp3['point'][i:i+self.batch_size]))
                self.current_epoch += 1
        
        elif self.mode == 'predict':
            cat = shapenet_category_to_id['chair']
            batch = self.cats_batches[cat]
            with h5py.File(os.path.join(self.image_path, '{}.h5'.format(cat)), 'r') as fi:
                with h5py.File(os.path.join(self.point_path_densepcr[16384], '{}.h5'.format(cat)), 'r') as fp3:
                    for i in range(36, batch * self.batch_size, self.batch_size):
                        if self.stopped:
                            break 
                        self.queue.put((fi['image'][i:i+self.batch_size].astype('float32') / 255.0, fp3['point'][i:i+self.batch_size]))


        else:
            for cat, batch in self.cats_batches.iteritems():
               with h5py.File(os.path.join(self.image_path, '{}.h5'.format(cat)), 'r') as fi:
                    with h5py.File(os.path.join(self.point_path_densepcr[16384], '{}.h5'.format(cat)), 'r') as fp3:
                        for i in range(0, batch * self.batch_size, self.batch_size):
                            if self.stopped:
                                break
                            self.queue.put((fi['image'][i:i+self.batch_size].astype('float32') / 255.0, fp3['point'][i:i+self.batch_size]))

    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()
	
    def shutdown(self):
        self.stopped = True
        while not self.queue.empty():
            self.queue.get()



if __name__ == '__main__':
    data = DataFetcher('test',batch_size = 1)
    data.start()
    image, point = data.fetch()
    # current = 0

    # #create white background
    # background = np.zeros((256,256,3), dtype = np.uint8)
    # background.fill(255)

    # # 1. image (obj rendering) 
    # img = image[current, ...] * 255
    # img = img.astype('uint8')
    # img += background
    # img = np.where(img > 255 , img - 255, img)
    # cv2.imwrite('{:0>4}.png'.format(current), img)

    # # 3. gt_rendering
    # gt_rendering = background
    # X, Y, Z = point.T
    # F = 284
    # h = (-Y)/(-Z)*F + 256/2.0
    # w = X/(-Z)*F + 256/2.0
    # # h = np.minimum(np.maximum(h, 0), 255)
    # # w = np.minimum(np.maximum(w, 0), 255)
    # gt_rendering[np.round(h).astype(int), np.round(w).astype(int), 0] = 0
    # gt_rendering[np.round(h).astype(int), np.round(w).astype(int), 2] = 0
    # cv2.imwrite('{:0>4}.jpg'.format(current), gt_rendering)
    data.shutdown()
