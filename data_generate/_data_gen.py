import cPickle as pickle
import numpy as np
from filecmp import dircmp
import os
import cv2
from progress.bar import IncrementalBar
import time
import h5py

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

def get_path():
    image_prefix = '/media/tree/data1/projects/AttentionBased/data/image_png'
    point_prefix = '/media/tree/data1/projects/AttentionBased/data/pointcloud/16384'
    image_output_folder = '/media/tree/backup/projects/AttentionBased/data/image'
    point_output_folder = '/media/tree/backup/projects/AttentionBased/data/pointcloud_16384'
    npz_output_folder = '/media/tree/backup/projects/AttentionBased/data/data'

    image_path = []
    point_path = []
    # common_path = []
    print('preparing data')
    for dir_top, subdir_cmps in dircmp(image_prefix,point_prefix).subdirs.items():
        # print(dir_top, subdir_cmps)
        for dir_bottom in subdir_cmps.common_dirs:
            # print('preparing')
            # common_path.append(os.path.join(dir_top, dir_bottom)
            for index in range(24):
                image_path.append(os.path.join(image_prefix, dir_top, dir_bottom, '{0:02d}.png'.format(int(index))))
                point_path.append(os.path.join(point_prefix, dir_top, dir_bottom, '{0:02d}.npy'.format(int(index))))
    print('data prepared')
    # print(image_path[0])
    # print(image_path[0].split('/')[-3])
    return image_path, point_path, image_output_folder, point_output_folder, npz_output_folder

def dump_data():
    image_path, point_path, image_output_folder, point_output_folder, npz_output_folder = get_path()
    bar =  IncrementalBar(max=len(image_path))
    count = 0
    batch_image = np.empty((24,256,256,3), dtype = np.uint8)
    batch_point = np.empty((24,16384,3), dtype = np.float32)
    for i in range(24):
    # for i in range(len(image_path)):
        image = cv2.imread(image_path[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_array = np.array(image)
        image_array = image_array.astype(np.uint8)
        point_array = np.load(point_path[i])
        point_array = point_array.astype(np.float32)
        # np.save(os.path.join(image_output_folder,"{}_{}_{}.npy".format(image_path[i].split('/')[-3], image_path[i].split('/')[-2], image_path[i].split('/')[-1].split('.')[0])), image_array)
        # np.save(os.path.join(point_output_folder,"{}_{}_{}.npy".format(image_path[i].split('/')[-3], image_path[i].split('/')[-2], image_path[i].split('/')[-1].split('.')[0])), point_array)
        
        batch_image[count, ...] = image_array
        batch_point[count, ...] = point_array
        count += 1

        while count == 24:
            count = 0
            np.save(os.path.join(image_output_folder,"{}_{}.npy".format(image_path[i].split('/')[-3], image_path[i].split('/')[-2])), batch_image)
            np.save(os.path.join(point_output_folder,"{}_{}.npy".format(image_path[i].split('/')[-3], image_path[i].split('/')[-2])), batch_point)
            np.savez_compressed(os.path.join(npz_output_folder,"{}_{}.npz".format(image_path[i].split('/')[-3], image_path[i].split('/')[-2])), image = batch_image, point = batch_point)
            # print()
            # print(batch_image.shape)
            # print(batch_image.dtype)
            # print(batch_image.size)
            # print(batch_point.shape)
            # print(batch_point.dtype)
            # print(batch_point.size)
            # load = np.load(os.path.join(npz_output_folder,"{}_{}.npz".format(image_path[i].split('/')[-3], image_path[i].split('/')[-2])))
            # load_image = load['image']
            # load_point = load['point']
            # print(load_image.shape)
            # print(load_image.dtype)
            # print(load_image.size)
            # print(load_point.shape)
            # print(load_point.dtype)
            # print(load_point.size)
            batch_image = np.empty((24,256,256,3), dtype = np.uint8)
            batch_point = np.empty((24,16384,3), dtype = np.float32)

        # with open(os.path.join(output_folder,'{}_{}_{}.pkl'.format(image_path[i].split('/')[-3], image_path[i].split('/')[-2], image_path[i].split('/')[-1].split('.')[0])),'w') as f:
        #     pickle.dump([image_array,point_array], f)
        bar.next()
    bar.finish()

def dump_single():
    image_path = '/media/tree/data1/projects/AttentionBased/data/image_png/04090263/2d203e283c13fd16494585aaf374e961/00.png'
    point_path = '/media/tree/data1/projects/AttentionBased/data/pointcloud/16384/04090263/2d203e283c13fd16494585aaf374e961/00.npy'
    output_folder = '/media/tree/data1/projects/AttentionBased/PSGN'
    pixel2mesh_path = '/media/tree/data1/projects/Pixel2Mesh/pixel2mesh/data/ShapeNetTrain/04090263_2d203e283c13fd16494585aaf374e961_00.dat'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_array = np.array(image)
    print(image_array.dtype)
    # image_array = image.astype('float32') / 255.0
    point_array = np.load(point_path)
    # print(point_array.shape)
    # np.savez(os.path.join(output_folder,"{}_{}_{}.npz".format(image_path.split('/')[-3], image_path.split('/')[-2], image_path.split('/')[-1].split('.')[0])), image = image_array, point = point_array)
    
    # pkl = pickle.load(open(pixel2mesh_path, 'rb'))
    # pixel_image = pkl[0]
    # pixel_label = pkl[1]
    # print(pixel_label.shape)

    # np.savez(os.path.join(output_folder,"pixel.npz"), image = pixel_image, point = pixel_label)

    # with open(os.path.join(output_folder,"{}_{}_{}.pkl".format(image_path.split('/')[-3], image_path.split('/')[-2], image_path.split('/')[-1].split('.')[0])), 'wb') as f:

    #     pickle.dump((image_array,point_array), f)
    # with open(os.path.join(output_folder,'{}_{}_{}.pkl'.format(image_path.split('/')[-3], image_path.split('/')[-2], image_path.split('/')[-1].split('.')[0])),'w') as f:
    #     pickle.dump([image_array,point_array], f, 2)

if __name__ == '__main__':
    print(time.strftime("%m-%d %H:%M:%S", time.localtime()))
    dump_data()
    print(time.strftime("%m-%d %H:%M:%S", time.localtime()))
        