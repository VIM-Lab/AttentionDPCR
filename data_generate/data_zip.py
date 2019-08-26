import numpy as np
import os
import cv2
from progress.bar import IncrementalBar
import time
import h5py
import json
import re
import sys

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


def dump_image_point():
    data_prefix = '/media/tree/data1/projects/AttentionBased/data'
    train_output_folder = '/media/tree/backup/projects/AttentionBased/data/train'
    test_output_folder = '/media/tree/backup/projects/AttentionBased/data/test'
    image_input_folder = 'image_256_256_12'
    point_input_folder = 'pointcloud_12/16384'
    image_output_folder = 'image_256_256_12'
    image_192_output_folder = 'image_192_256_12'
    point_output_folder = 'point_16384_12'
    image_number = 12

    with open('/media/tree/backup/projects/AttentionBased/data/train_models.json', 'r') as f:
        train_models_dict = json.load(f)

    with open('/media/tree/backup/projects/AttentionBased/data/test_models.json', 'r') as f:
        test_models_dict = json.load(f)

    cats = shapenet_category_to_id.values()
    for cat in cats:
        print(cat, 'starts at ', time.strftime("%m-%d %H:%M:%S", time.localtime()))
        print(cat, 'loading train_split!')
        train_image_models = []
        train_point_models = []
        train_img_path = []
        train_pt_path = []
        train_image_models.extend([os.path.join(data_prefix, image_input_folder, model) for model in train_models_dict[cat]])
        for each in train_image_models:
            for index in range(image_number):
                train_img_path.append(os.path.join(each, '{0:02d}.png'.format(int(index))))
        
        train_point_models.extend([os.path.join(data_prefix, point_input_folder, model) for model in train_models_dict[cat]])
        for each in train_point_models:
            for index in range(image_number):
                train_pt_path.append(os.path.join(each, '{0:02d}.npy'.format(int(index))))
                
        print(cat, 'train_split loaded!')

        train_image_save = h5py.File(os.path.join(train_output_folder, image_output_folder, '{}.h5'.format(cat)), mode = 'w')
        train_image_192_save = h5py.File(os.path.join(train_output_folder, image_192_output_folder, '{}.h5'.format(cat)), mode = 'w')
        # train_point_save = h5py.File(os.path.join(train_output_folder, point_output_folder, '{}.h5'.format(cat)), mode = 'w')
        
        train_img_shape = (len(train_img_path), 256, 256, 3)
        train_img_192_shape = (len(train_img_path), 192, 256, 3)
        train_pt_shape = (len(train_pt_path), 16384, 3)

        train_image_save.create_dataset('image', train_img_shape, np.uint8)
        train_image_192_save.create_dataset('image', train_img_192_shape, np.uint8)
        # train_point_save.create_dataset('point', train_pt_shape, np.float32)
        
        print(cat, 'saving train data at', time.strftime("%m-%d %H:%M:%S", time.localtime()))
        train_bar =  IncrementalBar(max=len(train_img_path))
        for i in range(len(train_img_path)):
            image_array, point_array , image_192_array = load_data(train_img_path[i], train_pt_path[i])
            train_image_save['image'][i, ...] = image_array
            train_image_192_save['image'][i, ...] = image_192_array
            # train_point_save['point'][i, ...] = point_array
            train_bar.next()
        train_bar.finish()
        print(cat, 'train data saved!')
        
        train_image_save.close()
        train_image_192_save.close()
        # train_point_save.close()

        print(cat, 'loading test_split!')
        test_image_models = []
        test_point_models = []
        test_img_path = []
        test_pt_path = []
        test_image_models.extend([os.path.join(data_prefix, image_input_folder, model) for model in test_models_dict[cat]])
        for each in test_image_models:
            for index in range(image_number):
                test_img_path.append(os.path.join(each, '{0:02d}.png'.format(int(index))))
        
        test_point_models.extend([os.path.join(data_prefix, point_input_folder, model) for model in test_models_dict[cat]])
        for each in test_point_models:
            for index in range(image_number):
                test_pt_path.append(os.path.join(each, '{0:02d}.npy'.format(int(index))))
        
        print(cat, 'test_split loaded!')

        test_image_save = h5py.File(os.path.join(test_output_folder, image_output_folder, '{}.h5'.format(cat)), mode = 'w')
        test_image_192_save = h5py.File(os.path.join(test_output_folder, image_192_output_folder, '{}.h5'.format(cat)), mode = 'w')
        # test_point_save = h5py.File(os.path.join(test_output_folder, point_output_folder, '{}.h5'.format(cat)), mode = 'w')
        
        test_img_shape = (len(test_img_path), 256, 256, 3)
        test_img_192_shape = (len(test_img_path), 192, 256, 3)
        test_pt_shape = (len(test_pt_path), 16384, 3)
        
        test_image_save.create_dataset('image', test_img_shape, np.uint8)
        test_image_192_save.create_dataset('image', test_img_192_shape, np.uint8)
        # test_point_save.create_dataset('point', test_pt_shape, np.float32)

        print(cat, 'saving test data at ', time.strftime("%m-%d %H:%M:%S", time.localtime()))
        test_bar =  IncrementalBar(max=len(test_img_path))
        for i in range(len(test_img_path)):
            image_array, point_array , image_192_array = load_data(test_img_path[i], test_pt_path[i])
            test_image_save['image'][i, ...] = image_array
            test_image_192_save['image'][i, ...] = image_192_array
            # test_point_save['point'][i, ...] = point_array
            test_bar.next()
        test_bar.finish()
        print(cat, 'test data saved!')
        
        print(cat, 'finished at ', time.strftime("%m-%d %H:%M:%S", time.localtime()))
        
        test_image_save.close()
        test_image_192_save.close()
        # test_point_save.close()

def load_data(image_path, point_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_192 = cv2.resize(image, (256,192))
    image_array = np.array(image)
    image_array = image_array.astype(np.uint8)
    image_192_array = np.array(image_192)
    image_192_array = image_192_array.astype(np.uint8)
    point_array = np.load(point_path)
    point_array = point_array.astype(np.float32)

    return image_array, point_array, image_192_array

# def load_single():
#     data_prefix = '/media/tree/data1/projects/AttentionBased/data'
#     with open('/media/tree/data1/projects/PointGAN/3d-lmnet/data/splits/train_models.json', 'r') as f:
#         train_models_dict = json.load(f)
#     train_image_models = []
#     train_img_path = []
#     train_image_models.extend([os.path.join(data_prefix, 'image_png', model) for model in train_models_dict['04090263']])
#     for each in train_image_models:
#         for index in range(24):
#             train_img_path.append(os.path.join(each, '{0:02d}.png'.format(int(index))))
#     for i in range(len(train_img_path)):
#         image = cv2.imread(train_img_path[i])
#         print(train_img_path[i])
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

if __name__ == '__main__':
    dump_image_point()
