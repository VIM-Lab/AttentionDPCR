import numpy as np
from tf_sampling import farthest_point_sample, gather_point
import tensorflow as tf
import os
import h5py
from progress.bar import IncrementalBar
from utils.datafetcher import DataFetcher
import time

shapenet_id_to_category = {
'02691156': 'airplane',
'02828884': 'bench',
'02933112': 'cabinet',
'02958343': 'car',
'03001627': 'chair',
'03211117': 'monitor',
'03636649': 'lamp',
'03691459': 'speaker',
'04090263': 'rifle',
'04256520': 'sofa',
'04379243': 'table',
'04401088': 'telephone',
'04530566': 'vessel'
}

def resample(pointcloud, npoint):
    new_pointcloud = gather_point(pointcloud, farthest_point_sample(npoint, pointcloud))
    return new_pointcloud

def resample_all():
    data_prefix = '/media/tree/data1/projects/AttentionBased/data'
    train_output_folder = '/media/tree/backup/projects/AttentionBased/data/train'
    test_output_folder = '/media/tree/backup/projects/AttentionBased/data/test'
    # point_output_folder = 'point_resample_point_num_12'
    point_output_folder = 'point_4096_12'
    batch_size = 12
    resample_point_num = 4096

    #GPU settings 90% memory usage
    config = tf.ConfigProto() 
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True

    data = DataFetcher('train', batch_size = batch_size, epoch = 1)
    data.setDaemon(True)
    data.start()

    pt = tf.placeholder(tf.float32, shape = (None, 16384, 3), name = 'point')

    with tf.Session(config = config) as sess:
        
        for cat, batch in data.cats_batches.iteritems():
            print('Resampling train {} starts at {}'.format(shapenet_id_to_category[cat], time.strftime("%m-%d %H:%M:%S", time.localtime())))
            train_point_save = h5py.File(os.path.join(train_output_folder, point_output_folder, '{}.h5'.format(cat)), mode = 'w')
            train_pt_shape = (batch*batch_size, resample_point_num, 3)
            train_point_save.create_dataset('point', train_pt_shape, np.float32)
            bar = IncrementalBar(max = batch)
            for i in range(batch):
                _image, point = data.fetch()
                pointcloud = resample(pt, resample_point_num)
                point_array = sess.run(pointcloud, feed_dict = {pt:point})
                # point_array = np.reshape(new_pt,(resample_point_num, 3))
                # np.savetxt(os.path.join(train_output_folder, point_output_folder, '{}_{}.xyz'.format(cat, i)), new_pt)
                train_point_save['point'][i*batch_size:i*batch_size+batch_size, ...] = point_array
                bar.next()
            bar.finish()
            train_point_save.close()
    data.shutdown()

    data = DataFetcher('test', batch_size = batch_size, epoch = 1)
    data.setDaemon(True)
    data.start()

    with tf.Session(config = config) as sess:
        # pt = tf.placeholder(tf.float32, shape = (None, 16384, 3), name = 'point')
        for cat, batch in data.cats_batches.iteritems():
            print('Resampling test {} starts at {}'.format(shapenet_id_to_category[cat], time.strftime("%m-%d %H:%M:%S", time.localtime())))
            test_point_save = h5py.File(os.path.join(test_output_folder, point_output_folder, '{}.h5'.format(cat)), mode = 'w')
            test_pt_shape = (batch*batch_size, resample_point_num, 3)
            test_point_save.create_dataset('point', test_pt_shape, np.float32)
            bar = IncrementalBar(max = batch)
            for i in range(batch):
                _image, point = data.fetch()
                pointcloud = resample(pt, resample_point_num)
                point_array = sess.run(pointcloud, feed_dict = {pt:point})
                # point_array = np.reshape(new_pt,(resample_point_num, 3))
                # np.savetxt(os.path.join(train_output_folder, point_output_folder, '{}_{}.xyz'.format(cat, i)), new_pt)
                test_point_save['point'][i*batch_size:i*batch_size+batch_size, ...] = point_array
                bar.next()
            bar.finish()
            test_point_save.close()
    data.shutdown()

if __name__ == '__main__':
    resample_all()
