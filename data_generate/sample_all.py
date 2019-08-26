import os
from filecmp import dircmp
from progress.bar import IncrementalBar
import numpy as np
from sample_single import sample_single

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

def get_path(cat_id):
    obj_prefix = os.path.join('/media/tree/data1/data/ShapeNetCore.v1',cat_id)
    view_prefix = os.path.join('/media/tree/data1/projects/PointGAN/3d-lmnet/data/ShapeNetRendering',cat_id)
    output_prefix = os.path.join('/media/tree/data1/projects/AttentionBased/data/pointcloud_12/16384',cat_id)
    view_path = '/media/tree/data1/projects/AttentionBased/PSGN/data_generate/rendering_metadata.txt'
    if not os.path.isdir(output_prefix):
        os.makedirs(output_prefix)
    obj_path = []
    # view_path = []
    output_folder = []

    for i in dircmp(obj_prefix,view_prefix).common_dirs:
        obj_path.append(os.path.join(obj_prefix, i, 'model.obj'))
        # view_path.append(os.path.join(view_prefix, i, 'rendering/rendering_metadata.txt'))
        output_folder.append(os.path.join(output_prefix, i))
        
    for i in output_folder:
        if not os.path.isdir(i):
            os.makedirs(i)
    return obj_path, view_path, output_folder

def sample_all():
    for cat, cat_id in shapenet_category_to_id.items():
        obj_path, view_path, output_folder = get_path(cat_id)
        print('Sampling %d pointclouds for cat %s' % (len(obj_path),cat_id))
        bar =  IncrementalBar(max=len(obj_path))
        for i in range(len(obj_path)):
            bar.next()
            sample_single(obj_path[i], view_path, output_folder[i])
        bar.finish()
    print('All cats sampled!')

if __name__ == '__main__':
    sample_all()
