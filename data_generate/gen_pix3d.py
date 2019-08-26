from obj_io import parse_obj_file
from sample_single import sample_faces
import os
import json
import numpy as np
from progress.bar import IncrementalBar
data_dir_imgs = '/media/tree/data1/data/pix3d'

def get_pix3d_models():

	with open(os.path.join(data_dir_imgs, 'pix3d.json'), 'r') as f:
		models_dict = json.load(f)
	models = []

	cats = ['chair','sofa','table']

	
	# Check for truncation and occlusion before adding a model to the evaluation list
	for d in models_dict:
		if d['category'] in cats:
			if not d['truncated'] and not d['occluded'] and not d['slightly_occluded']:
				models.append(d)

	print 'Total models = {}\n'.format(len(models))
	return models

def sample_single(obj_path):
    # 1 sampling
    with open(obj_path,'r') as f:
        vertices, faces = parse_obj_file(f)[:2]
    sample = sample_faces(vertices, faces, 16384)
    position = sample * 0.57
    npy_path = obj_path.split('.')[0]
    np.save(npy_path, position)

def sample_all(models):
    bar =  IncrementalBar(max=len(models))
    for i in range(len(models)):
        sample_single(os.path.join(data_dir_imgs, models[i]['model']))
        bar.next()
    bar.finish()

if __name__ == '__main__':
    pix3d_js = get_pix3d_models()
    # for i in pix3d_js:
    #     print i['model']
    sample_all(pix3d_js)
