import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
# import seaborn as sns
import ctypes as ct

dll = np.ctypeslib.load_library('render_balls_so','./utils')

def get_path(model_name, cat):

	vis_dir = '/media/tree/backup/projects/AttentionBased_vis'
	image_dir = '/media/tree/backup/projects/AttentionBased_vis/image'
	gt_point_dir = '/media/tree/backup/projects/AttentionBased_vis/ground_truth_point'
	gt_rendering_dir = '/media/tree/backup/projects/AttentionBased_vis/groud_truth_rendring'

	pred_point_dir = os.path.join(vis_dir, 'predicted_point_{}'.format(model_name))
	if not os.path.isdir(pred_point_dir):
		os.makedirs(pred_point_dir)
	
	pred_rendering_dir = os.path.join(vis_dir, 'predicted_rendering_{}'.format(model_name))
	if not os.path.isdir(pred_rendering_dir):
		os.makedirs(pred_rendering_dir)
	
	image_cat_dir = os.path.join(image_dir, cat)
	if not os.path.isdir(image_cat_dir):
		os.makedirs(image_cat_dir)
	
	gt_point_cat_dir = os.path.join(gt_point_dir, cat)
	if not os.path.isdir(gt_point_cat_dir):
		os.makedirs(gt_point_cat_dir)
	
	gt_rendering_cat_dir = os.path.join(gt_rendering_dir, cat)
	if not os.path.isdir(gt_rendering_cat_dir):
		os.makedirs(gt_rendering_cat_dir)
	
	pred_point_cat_dir = os.path.join(pred_point_dir, cat)
	if not os.path.isdir(pred_point_cat_dir):
		os.makedirs(pred_point_cat_dir)
	
	pred_rendering_cat_dir = os.path.join(pred_rendering_dir, cat)
	if not os.path.isdir(pred_rendering_cat_dir):
		os.makedirs(pred_rendering_cat_dir)
	
	# feature_map_dir = os.path.join(vis_dir, 'feature_map_{}'.format(model_name))
	# if not os.path.isdir(feature_map_dir):
	# 	os.makedirs(feature_map_dir)

	# feature_map_cat_dir = os.path.join(feature_map_dir, cat)
	# if not os.path.isdir(feature_map_cat_dir):
	# 	os.makedirs(feature_map_cat_dir)

	return image_cat_dir, gt_point_cat_dir, gt_rendering_cat_dir, pred_point_cat_dir, pred_rendering_cat_dir

def get_rendering(point, ballradius=2, background=(255,255,255), image_size=256):
	point=point-point.mean(axis=0)
	radius=((point**2).sum(axis=-1)**0.5).max()
	point/=(radius*2.2)/image_size

	c0=np.zeros((len(point),),dtype='float32')+242 #G
	c1=np.zeros((len(point),),dtype='float32')+248 #R
	c2=np.zeros((len(point),),dtype='float32')+220 #B

	c0=np.require(c0,'float32','C')
	c1=np.require(c1,'float32','C')
	c2=np.require(c2,'float32','C')

	show=np.zeros((image_size,image_size,3),dtype='uint8')
	def render():
		npoint = point + [image_size/2,image_size/2,0]
		# npoint = point
		ipoint = npoint.astype('int32')
		show[:]=background
		dll.render_ball(
			ct.c_int(show.shape[0]),
			ct.c_int(show.shape[1]),
			show.ctypes.data_as(ct.c_void_p),
			ct.c_int(ipoint.shape[0]),
			ipoint.ctypes.data_as(ct.c_void_p),
			c0.ctypes.data_as(ct.c_void_p),
			c1.ctypes.data_as(ct.c_void_p),
			c2.ctypes.data_as(ct.c_void_p),
			ct.c_int(ballradius)
		)

	render()
	return show

#visualize all images, only have to run once
def visualize_img_all(image, point, predicted_point, iteration, batch_size, model_name, cat):
	#get path
	image_dir, gt_point_dir, gt_rendering_dir, pred_point_dir, pred_rendering_dir = get_path(model_name, cat)
	
	for i in range(batch_size):

		current = iteration * batch_size + i
		
		#create white background
		background = np.zeros((256,256,3), dtype = np.uint8)
		background.fill(255)
		
		# 1. image (obj rendering) 
		img = image[i, ...] * 255
		img = img.astype('uint8')
		img += background
		img = np.where(img > 255 , img - 255, img)
		cv2.imwrite(os.path.join(image_dir, '{:0>5}.png'.format(current)), img)

		# 2. gt_point
		gt_pt = point[i, ...]
		
		# np.savetxt(os.path.join(gt_point_dir, '{:0>5}.xyz'.format(current)), gt_pt)
		
		gt_pt = np.expand_dims(gt_pt,axis = 0)

		# # 3. gt_rendering (point)
		# gt_rendering = background
		# X, Y, Z = gt_pt.T
		# F = 284
		# h = (-Y)/(-Z)*F + 256/2.0
		# w = X/(-Z)*F + 256/2.0
		# h = np.minimum(np.maximum(h, 0), 255)
		# w = np.minimum(np.maximum(w, 0), 255)
		# gt_rendering[np.round(h).astype(int), np.round(w).astype(int), 0] = 0
		# gt_rendering[np.round(h).astype(int), np.round(w).astype(int), 2] = 0
		# cv2.imwrite(os.path.join(gt_rendering_dir, '{:0>5}.png'.format(current)), gt_rendering)

		# 3. gt_rendering (ball)
		X, Y, Z = gt_pt.T
		gt_pt_t = np.concatenate([-Y,X,Z],1)
		gt_rendering = get_rendering(np.vstack(gt_pt_t))
		cv2.imwrite(os.path.join(gt_rendering_dir, '{:0>5}.png'.format(current)), gt_rendering)

		# 4. pred_point
		pred_point = predicted_point[i, ...]
		
		# np.savetxt(os.path.join(pred_point_dir, '{:0>5}.xyz'.format(current)), pred_point)

		pred_point = np.expand_dims(pred_point,0)

		# # 5. pred_rendering (point)
		# pred_rendering = background
		# X, Y, Z = pred_point.T
		# F = 284
		# h = (-Y)/(-Z)*F + 256/2.0
		# w = X/(-Z)*F + 256/2.0
		# h = np.minimum(np.maximum(h, 0), 255)
		# w = np.minimum(np.maximum(w, 0), 255)
		# pred_rendering[np.round(h).astype(int), np.round(w).astype(int), 0] = 0
		# pred_rendering[np.round(h).astype(int), np.round(w).astype(int), 2] = 0
		# cv2.imwrite(os.path.join(pred_rendering_dir, '{:0>5}.png'.format(current)), pred_rendering)

		# 5. pred_rendering (ball)
		X, Y, Z = pred_point.T
		pred_point_t = np.concatenate([-Y,X,Z],1)
		pred_rendering = get_rendering(np.vstack(pred_point_t))
		cv2.imwrite(os.path.join(pred_rendering_dir, '{:0>5}.png'.format(current)), pred_rendering)

def visualize_input_pix3d(image, iteration, batch_size, model_name, cat):
	image_dir = '/media/tree/backup/projects/AttentionBased_vis/image_pix3d_origin' 
	image_dir = os.path.join(image_dir, cat)
	if not os.path.isdir(image_dir):
		os.makedirs(image_dir)
	for i in range(batch_size):

		current = iteration * batch_size + i
		
		#create white background
		background = np.zeros((256,256,3), dtype = np.uint8)
		background.fill(255)
		
		# 1. image (obj rendering) 
		img = image[i, ...] * 255
		img = img.astype('uint8')
		img += background
		img = np.where(img > 255 , img - 255, img)
		cv2.imwrite(os.path.join(image_dir, '{:0>5}.png'.format(current)), img)
	

def visualize_img_all_pix3d(image, point, predicted_point, iteration, batch_size, model_name, cat):
	#get path
	image_dir, gt_point_dir, gt_rendering_dir, pred_point_dir, pred_rendering_dir = get_path(model_name, cat)
	
	for i in range(batch_size):

		current = iteration * batch_size + i
		
		#create white background
		background = np.zeros((256,256,3), dtype = np.uint8)
		background.fill(255)
		
		# 1. image (obj rendering) 
		# img = image[i, ...] * 255
		# img = img.astype('uint8')
		# img += background
		# img = np.where(img > 255 , img - 255, img)
		# cv2.imwrite(os.path.join(image_dir, '{:0>5}.png'.format(current)), img)

		# 2. gt_point
		gt_pt = point[i, ...]
		
		# np.savetxt(os.path.join(gt_point_dir, '{:0>5}.xyz'.format(current)), gt_pt)
		
		gt_pt = np.expand_dims(gt_pt,axis = 0)

		# # 3. gt_rendering (point)
		# gt_rendering = background
		# X, Y, Z = gt_pt.T
		# F = 284
		# h = (-Y)/(-Z)*F + 256/2.0
		# w = X/(-Z)*F + 256/2.0
		# h = np.minimum(np.maximum(h, 0), 255)
		# w = np.minimum(np.maximum(w, 0), 255)
		# gt_rendering[np.round(h).astype(int), np.round(w).astype(int), 0] = 0
		# gt_rendering[np.round(h).astype(int), np.round(w).astype(int), 2] = 0
		# cv2.imwrite(os.path.join(gt_rendering_dir, '{:0>5}.png'.format(current)), gt_rendering)

		# 3. gt_rendering (ball)
		X, Y, Z = gt_pt.T
		gt_pt_t = np.concatenate([-Y,X,Z],1)
		gt_rendering = get_rendering(np.vstack(gt_pt_t))
		cv2.imwrite(os.path.join(gt_rendering_dir, '{:0>5}.png'.format(current)), gt_rendering)

		# 4. pred_point
		pred_point = predicted_point[i, ...]
		
		# np.savetxt(os.path.join(pred_point_dir, '{:0>5}.xyz'.format(current)), pred_point)

		pred_point = np.expand_dims(pred_point,0)

		# # 5. pred_rendering (point)
		# pred_rendering = background
		# X, Y, Z = pred_point.T
		# F = 284
		# h = (-Y)/(-Z)*F + 256/2.0
		# w = X/(-Z)*F + 256/2.0
		# h = np.minimum(np.maximum(h, 0), 255)
		# w = np.minimum(np.maximum(w, 0), 255)
		# pred_rendering[np.round(h).astype(int), np.round(w).astype(int), 0] = 0
		# pred_rendering[np.round(h).astype(int), np.round(w).astype(int), 2] = 0
		# cv2.imwrite(os.path.join(pred_rendering_dir, '{:0>5}.png'.format(current)), pred_rendering)

		# 5. pred_rendering (ball)
		X, Y, Z = pred_point.T
		pred_point_t = np.concatenate([-Y,X,Z],1)
		pred_rendering = get_rendering(np.vstack(pred_point_t))
		cv2.imwrite(os.path.join(pred_rendering_dir, '{:0>5}.png'.format(current)), pred_rendering)


def visualize_img(predicted_point, iteration, batch_size, model_name, cat):
	#get path
	_image_dir, _gt_point_dir, _gt_rendering_dir, pred_point_dir, pred_rendering_dir = get_path(model_name, cat)
	
	for i in range(batch_size):
		
		current = iteration * batch_size + i
		
		#create white background
		background = np.zeros((256,256,3), dtype = np.uint8)
		background.fill(255)
		
		# 1. pred_point
		pred_point = predicted_point[i, ...]
		# np.savetxt(os.path.join(pred_point_dir, '{:0>5}.xyz'.format(current)), pred_point)
		pred_point = np.expand_dims(pred_point,0)
		# # 2. pred_rendering (point)
		# pred_rendering = background
		# X, Y, Z = pred_point.T
		# F = 284
		# h = (-Y)/(-Z)*F + 256/2.0
		# w = X/(-Z)*F + 256/2.0
		# h = np.minimum(np.maximum(h, 0), 255)
		# w = np.minimum(np.maximum(w, 0), 255)
		# pred_rendering[np.round(h).astype(int), np.round(w).astype(int), 0] = 0
		# pred_rendering[np.round(h).astype(int), np.round(w).astype(int), 2] = 0
		# cv2.imwrite(os.path.join(pred_rendering_dir, '{:0>5}.png'.format(current)), pred_rendering)

		# 2. pred_rendering (ball)
		X, Y, Z = pred_point.T
		pred_point_t = np.concatenate([-Y,X,Z],1)
		pred_rendering = get_rendering(np.vstack(pred_point_t))
		cv2.imwrite(os.path.join(pred_rendering_dir, '{:0>5}.png'.format(current)), pred_rendering)

def visualize_img_PSGN(predicted_point, iteration, batch_size, model_name, cat):
	#get path
	_image_dir, _gt_point_dir, _gt_rendering_dir, pred_point_dir, pred_rendering_dir = get_path(model_name, cat)
	
	for i in range(batch_size):
		
		current = iteration * batch_size + i
		
		#create white background
		background = np.zeros((192,256,3), dtype = np.uint8)
		background.fill(255)
		
		# 1. pred_point
		pred_point = predicted_point[i, ...]
		np.savetxt(os.path.join(pred_point_dir, '{:0>5}.xyz'.format(current)), pred_point)
		pred_point = np.expand_dims(pred_point,0)
		# # 2. pred_rendering (point)
		# pred_rendering = background
		# X, Y, Z = pred_point.T
		# F = 284
		# h = (-Y)/(-Z)*F + 256/2.0
		# w = X/(-Z)*F + 256/2.0
		# h = np.minimum(np.maximum(h, 0), 255)
		# w = np.minimum(np.maximum(w, 0), 255)
		# pred_rendering[np.round(h).astype(int), np.round(w).astype(int), 0] = 0
		# pred_rendering[np.round(h).astype(int), np.round(w).astype(int), 2] = 0
		# cv2.imwrite(os.path.join(pred_rendering_dir, '{:0>5}.png'.format(current)), pred_rendering)

		# 2. pred_rendering (ball)
		X, Y, Z = pred_point.T
		pred_point_t = np.concatenate([-Y,X,Z],1)
		pred_rendering = get_rendering(np.vstack(pred_point_t))
		cv2.imwrite(os.path.join(pred_rendering_dir, '{:0>5}.png'.format(current)), pred_rendering)


def visualize_img_p2p(point, predicted_point, iteration, batch_size, model_name, cat):
	#get path
	_image_dir, gt_point_dir, gt_rendering_dir, pred_point_dir, pred_rendering_dir = get_path(model_name, cat)
	
	for i in range(batch_size):

		current = iteration * batch_size + i
		
		#create white background
		background = np.zeros((256,256,3), dtype = np.uint8)
		background.fill(255)
		
		# 2. gt_point
		gt_pt = point[i, ...]
		
		np.savetxt(os.path.join(gt_point_dir, '{:0>5}.xyz'.format(current)), gt_pt)
		
		gt_pt = np.expand_dims(gt_pt,axis = 0)

		# 3. gt_rendering (ball)
		X, Y, Z = gt_pt.T
		gt_pt_t = np.concatenate([-Y,X,Z],1)
		gt_rendering = get_rendering(np.vstack(gt_pt_t))
		cv2.imwrite(os.path.join(gt_rendering_dir, '{:0>5}.png'.format(current)), gt_rendering)


		# 1. pred_point
		pred_point = predicted_point[i, ...]
		np.savetxt(os.path.join(pred_point_dir, '{:0>5}.xyz'.format(current)), pred_point)
		pred_point = np.expand_dims(pred_point,0)

		# 2. pred_rendering (ball)
		X, Y, Z = pred_point.T
		pred_point_t = np.concatenate([-Y,X,Z],1)
		pred_rendering = get_rendering(np.vstack(pred_point_t))
		cv2.imwrite(os.path.join(pred_rendering_dir, '{:0>5}.png'.format(current)), pred_rendering)


def visualize_feature_map_sum(feature_batch, iteration, batch_size, model_name, cat, name):

	vis_dir = '/media/tree/backup/projects/AttentionBased_vis'

	feature_map_dir = os.path.join(vis_dir, 'feature_map_{}'.format(model_name))
	if not os.path.isdir(feature_map_dir):
		os.makedirs(feature_map_dir)

	feature_map_cat_dir = os.path.join(feature_map_dir, cat)
	if not os.path.isdir(feature_map_cat_dir):
		os.makedirs(feature_map_cat_dir)
	
	
	for i in range(batch_size):

		current = iteration * batch_size + i
		feature_map = feature_batch[i, ...]

		# num_pic = feature_map.shape[2]

		# for i in range(0, num_pic):
		# 	feature_map_split = feature_map[:, :, i]
		# 	feature_map_combination.append(feature_map_split)

		# feature_map_sum = sum(one for one in feature_map_combination)

		feature_map = np.mean(feature_map, axis = 2)
		# for i in range(feature_map.shape[2]):
		# 	feature_map_one = feature_map[:,:,i]
		# 	plt.figure(figsize=(1, 1))
		# 	# ax = sns.heatmap(feature_map_sum)
		# 	plt.imshow(feature_map_one, norm = plt.Normalize(0,1))
		# 	plt.axis('off')
		# 	plt.gca().xaxis.set_major_locator(plt.NullLocator())
		# 	plt.gca().yaxis.set_major_locator(plt.NullLocator())
		# 	plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
		# 	plt.margins(0,0)
		# 	plt.savefig(os.path.join(feature_map_cat_dir, '{:0>4}_{}_{}.png'.format(current, name, i)))
		
		# feature_map = feature_map[:,:,26]
		# # feature_map_sum = (feature_map_sum - np.min(feature_map_sum))/(np.max(feature_map_sum)-np.min(feature_map_sum))
		# # np.savetxt(os.path.join(feature_map_cat_dir, '{:0>4}_{}.txt'.format(current, name)),feature_map_sum)
		# # feature_map_sum = preprocessing.scale(feature_map_sum)
		# # min_max_scaler = MinMaxScaler()
		# # feature_map_sum = min_max_scaler.fit_transform(feature_map_sum)

		# # feature_map_sum = feature_map[:,:,0:3]
		# # cv2.imwrite(os.path.join(feature_map_cat_dir, '{:0>4}_{}.png'.format(current, name)), feature_map_one)

		plt.figure(figsize=(1, 1))
		# ax = sns.heatmap(feature_map_sum)
		plt.imshow(feature_map, norm = plt.Normalize(0,1))
		plt.axis('off')
		plt.gca().xaxis.set_major_locator(plt.NullLocator())
		plt.gca().yaxis.set_major_locator(plt.NullLocator())
		plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
		plt.margins(0,0)
		plt.savefig(os.path.join(feature_map_cat_dir, '{:0>4}_{}.png'.format(current, name)),dpi=256)
		plt.close('all')

def visualize_feature_map(feature_map_list, iteration, batch_size, model_name, cat):
	visualize_feature_map_sum(feature_map_list[0], iteration, batch_size, model_name, cat, 'before_sam1')
	visualize_feature_map_sum(feature_map_list[1], iteration, batch_size, model_name, cat, 'sam1')
	visualize_feature_map_sum(feature_map_list[2], iteration, batch_size, model_name, cat, 'after_sam1')
	visualize_feature_map_sum(feature_map_list[3], iteration, batch_size, model_name, cat, 'before_sam2')
	visualize_feature_map_sum(feature_map_list[4], iteration, batch_size, model_name, cat, 'sam2')
	visualize_feature_map_sum(feature_map_list[5], iteration, batch_size, model_name, cat, 'after_sam2')
	visualize_feature_map_sum(feature_map_list[6], iteration, batch_size, model_name, cat, 'before_sam3')
	visualize_feature_map_sum(feature_map_list[7], iteration, batch_size, model_name, cat, 'sam3')
	visualize_feature_map_sum(feature_map_list[8], iteration, batch_size, model_name, cat, 'after_sam3')
