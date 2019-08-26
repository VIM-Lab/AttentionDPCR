import os
import tensorflow as tf
import numpy as np
import tflearn
from metric.tf_nndistance import nn_distance
from metric.tf_approxmatch import approx_match, match_cost
from layer.encoders_decoders import *
from layer.pointnet_util import pointnet_sa_module, pointnet_sa_module_msg
from pu_util import *
from layer.basic_layers import ResidualBlock
from layer.attention_module import AttentionModule

def scale(gt, pred): #pr->[-0.5,0.5], gt->[-0.5,0.5]
	'''
	Scale the input point clouds between [-max_length/2, max_length/2]
	'''

	# Calculate min and max along each axis x,y,z for all point clouds in the batch
	min_gt = tf.convert_to_tensor([tf.reduce_min(gt[:,:,i], axis=1) for i in xrange(3)]) #(3, B)
	max_gt = tf.convert_to_tensor([tf.reduce_max(gt[:,:,i], axis=1) for i in xrange(3)]) #(3, B)
	min_pr = tf.convert_to_tensor([tf.reduce_min(pred[:,:,i], axis=1) for i in xrange(3)]) #(3, B)
	max_pr = tf.convert_to_tensor([tf.reduce_max(pred[:,:,i], axis=1) for i in xrange(3)]) #(3, B)

	# Calculate x,y,z dimensions of bounding cuboid
	length_gt = tf.abs(max_gt - min_gt) #(3, B)
	length_pr = tf.abs(max_pr - min_pr) #(3, B)

	# Calculate the side length of bounding cube (maximum dimension of bounding cuboid)
	# Then calculate the delta between each dimension of the bounding cuboid and the side length of bounding cube
	diff_gt = tf.reduce_max(length_gt, axis=0, keep_dims=True) - length_gt #(3, B)
	diff_pr = tf.reduce_max(length_pr, axis=0, keep_dims=True) - length_pr #(3, B)

	# Pad the xmin, xmax, ymin, ymax, zmin, zmax of the bounding cuboid to match the cuboid side length
	new_min_gt = tf.convert_to_tensor([min_gt[i,:] - diff_gt[i,:]/2. for i in xrange(3)]) #(3, B)
	new_max_gt = tf.convert_to_tensor([max_gt[i,:] + diff_gt[i,:]/2. for i in xrange(3)]) #(3, B)
	new_min_pr = tf.convert_to_tensor([min_pr[i,:] - diff_pr[i,:]/2. for i in xrange(3)]) #(3, B)
	new_max_pr = tf.convert_to_tensor([max_pr[i,:] + diff_pr[i,:]/2. for i in xrange(3)]) #(3, B)

	# Compute the side length of bounding cube
	size_pr = tf.reduce_max(length_pr, axis=0) #(B,)
	size_gt = tf.reduce_max(length_gt, axis=0) #(B,)

	# Calculate scaling factor according to scaled cube side length (here = 2.)
	scaling_factor_gt = 1. / size_gt #(B,)
	scaling_factor_pr = 1. / size_pr #(B,)

	# Calculate the min x,y,z coordinates for the scaled cube (here = (-1., -1., -1.))
	box_min = tf.ones_like(new_min_gt) * -0.5 #(3, B)

	# Calculate the translation adjustment factor to match the minimum coodinates of the scaled cubes
	adjustment_factor_gt = box_min - scaling_factor_gt * new_min_gt #(3, B)
	adjustment_factor_pr = box_min - scaling_factor_pr * new_min_pr #(3, B)

	# Perform scaling then translation to the point cloud ? verify this
	pred_scaled = tf.transpose((tf.transpose(pred) * scaling_factor_pr)) + tf.reshape(tf.transpose(adjustment_factor_pr), (-1,1,3))
	gt_scaled   = tf.transpose((tf.transpose(gt) * scaling_factor_gt)) + tf.reshape(tf.transpose(adjustment_factor_gt), (-1,1,3))

	return gt_scaled, pred_scaled


class Dense(object):
	def __init__(self, sess, mode, batch_size):
		self.sess = sess
		self.mode = mode

		self.pointclouds_input = tf.placeholder(tf.float32, shape=(batch_size, 1024, 3), name='pointclouds_input')
		self.pointclouds_gt = tf.placeholder(tf.float32, shape=(batch_size, 16384, 3), name='pointclouds_gt')
		
		self.gt = None
		self.pred = None
		self.mid = None

		self.train_loss = 0
		self.cd_loss = 0
		self.emd_loss = 0

		self.optimizer = None
		self.opt_op = None

		
		self.build()
		#initialize saver after build graph
		self.saver = tf.train.Saver(max_to_keep=50)
	
	def build(self):
		if self.mode == 'train':
			self.build_graph()
			# self.build_optimizer()
			self.build_optimizer_multi()
		elif self.mode == 'test':
			self.build_graph(is_training=False)
			self.build_loss_calculater()
		else:
			self.build_graph(is_training=False)
	
	def build_optimizer(self):
		print('Building chamfer distance loss optimizer...')
		self.optimizer = tf.train.AdamOptimizer(3e-5)
		dist1,idx1,dist2,idx2 = nn_distance(self.gt, self.pred)
		self.train_loss = (tf.reduce_mean(dist1) + 0.55 * tf.reduce_mean(dist2)) * 10000
		self.cd_loss = tf.reduce_mean(dist1) + tf.reduce_mean(dist2)
		self.opt_op = self.optimizer.minimize(self.train_loss)
	
	def build_optimizer_multi(self):
		print('Building chamfer distance loss optimizer...')
		self.optimizer = tf.train.AdamOptimizer(3e-5)
		dist1,idx1,dist2,idx2 = nn_distance(self.gt, self.mid)
		train_loss = (tf.reduce_mean(dist1) + tf.reduce_mean(dist2)) * 10000
		dist1,idx1,dist2,idx2 = nn_distance(self.gt, self.pred)
		self.train_loss = train_loss + (tf.reduce_mean(dist1) + tf.reduce_mean(dist2)) * 10000
		self.cd_loss = tf.reduce_mean(dist1) + tf.reduce_mean(dist2)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.opt_op = self.optimizer.minimize(self.train_loss)

	def build_loss_calculater(self):
		tflearn.is_training(False, session=self.sess)
		if self.mode == 'test':
			self.gt, self.pred = scale(self.gt, self.pred)
		#cd
		dist1,idx1,dist2,idx2 = nn_distance(self.gt, self.pred)
		self.cd_loss = tf.reduce_mean(dist1) + tf.reduce_mean(dist2)
		#emd
		match = approx_match(self.pred, self.gt)
		# self.emd_loss = tf.reduce_mean(match_cost(self.gt, self.pred, match)) / float(tf.shape(self.pred)[1])
		self.emd_loss = tf.reduce_mean(match_cost(self.pred, self.gt, match))


	def build_graph(self, is_training=True):
		print('Building dense network...')
		x = self.pointclouds_input
		# y = self.pointclouds_gt
		with tf.variable_scope('dense'):
			mid_x, x  = pu_dense_dgcnn(x, is_training)
			# x  = pu_dense_dgcnn(x, is_training)
			# x = pu_dense(x, is_training)

		total_param_num = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
		print("===Total number of parameters: %d===" % total_param_num)

		self.pred = x
		self.mid = mid_x
		self.gt = self.pointclouds_gt

		
	def train(self, image, point):
		_, train_loss, cd_loss = self.sess.run([self.opt_op, self.train_loss, self.cd_loss], feed_dict = {self.pointclouds_input : image, self.pointclouds_gt : point})
		return train_loss, cd_loss
	
	def test(self, image, point):
		cd, emd = self.sess.run([self.cd_loss, self.emd_loss], feed_dict = {self.pointclouds_input : image, self.pointclouds_gt : point})
		return cd, emd
		# cd = self.sess.run([self.cd_loss], feed_dict = {self.pointclouds_input : image, self.pointclouds_gt : point})
		# return cd
	
	def predict(self, image):
		predicted_pointcloud = self.sess.run(self.pred, feed_dict = {self.pointclouds_input : image})
		return predicted_pointcloud

class Finetune(object):
	def __init__(self, sess, mode, batch_size):
		self.attention_module = AttentionModule()
		self.residual_block = ResidualBlock()
		self.sess = sess
		self.mode = mode
		self.batch_size = batch_size
		self.hierarchies = [1024,16384]
		self.pcl_gt = {}
		self.out = {}
		self.mid = None
		self.mid_loss = 0
		self.cd_loss_pu = {}
		self.emd_loss_pu = {}
		self.image = tf.placeholder(tf.float32, shape = (self.batch_size, 256, 256, 3), name = 'image')
		for stage in self.hierarchies:
			self.pcl_gt[stage] = tf.placeholder(tf.float32, shape=(self.batch_size, stage, 3), name='pcl_gt_%d'%stage)
			self.cd_loss_pu[stage] = 0
			self.emd_loss_pu[stage] = 0

		self.train_loss = 0
		self.cd_loss = 0
		self.emd_loss = 0
		self.pred = None
		self.gt = None
		self.optimizer = None
		self.opt_op = None

		self.build()
		#initialize saver after build graph
		self.build_saver()
	
	def build(self):
		if self.mode == 'train':
			self.build_graph()
			self.build_optimizer()
		elif self.mode == 'test':
			self.build_graph(is_training=False)
			self.build_loss_calculater()
		else:
			self.build_graph(is_training=False)
	
	def build_optimizer(self):
		print('Building chamfer distance loss and emd loss optimizer...')
		self.optimizer = tf.train.AdamOptimizer(3e-5)
		dist1 = {}
		dist2 = {}
		match = {}

		dist1_mid,idx1,dist2_mid,idx2 = nn_distance(self.pcl_gt[16384], self.mid)
		self.mid_loss = (tf.reduce_mean(dist1_mid) + tf.reduce_mean(dist2_mid))

		for stage in self.hierarchies:
			dist1[stage],_idx1,dist2[stage],_idx2 = nn_distance(self.pcl_gt[stage], self.out[stage])
			self.cd_loss_pu[stage] = tf.reduce_mean(dist1[stage]) + tf.reduce_mean(dist2[stage])
			if stage == 1024:
				match[stage] = approx_match(self.pcl_gt[stage], self.out[stage])
				self.emd_loss_pu[stage] = tf.reduce_mean(match_cost(self.pcl_gt[stage], self.out[stage], match[stage]))
		self.train_loss = 0.5*(self.cd_loss_pu[1024]+self.emd_loss_pu[1024]/1000)+self.mid_loss+self.cd_loss_pu[16384]
		self.cd_loss = self.cd_loss_pu[16384]
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.opt_op = self.optimizer.minimize(self.train_loss)

	def build_loss_calculater(self):
		tflearn.is_training(False, session=self.sess)
		if self.mode == 'test':
			self.gt, self.pred = scale(self.gt, self.pred)
		#cd
		dist1,idx1,dist2,idx2 = nn_distance(self.gt, self.pred)
		self.cd_loss = tf.reduce_mean(dist1) + tf.reduce_mean(dist2)
		#emd
		match = approx_match(self.gt, self.pred)
		# self.emd_loss = tf.reduce_mean(match_cost(self.gt, self.pred, match)) / float(tf.shape(self.pred)[1])
		self.emd_loss = tf.reduce_mean(match_cost(self.gt, self.pred, match))

	def build_saver(self):
		sparse_vars = [var for var in tf.global_variables() if 'sparse' in var.name]
		dense_vars = [var for var in tf.global_variables() if 'dense' in var.name]

		self.saver = tf.train.Saver(max_to_keep=40)
		self.saver_sparse = tf.train.Saver(var_list=sparse_vars)
		self.saver_dense = tf.train.Saver(var_list=dense_vars, max_to_keep=20)

	def build_graph(self, is_training=True):
		print('Building finetune network...')
		x = self.image
		with tf.variable_scope('sparse'):
			self.out[1024] = self.attention(x,is_training)

		with tf.variable_scope('dense'):
			mid_x, self.out[16384]  = pu_dense_dgcnn(self.out[1024], is_training)
		self.pred = self.out[16384]
		self.mid = mid_x
		self.gt = self.pcl_gt[16384]
		
		total_param_num = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
		print("===Total number of parameters: %d===" % total_param_num)
		
	def train(self, image, point1, point2):
		_, train_loss, cd_loss = self.sess.run([self.opt_op, self.train_loss, self.cd_loss], feed_dict = {self.image : image, self.pcl_gt[1024] : point1, self.pcl_gt[16384] : point2})
		return train_loss, cd_loss
	
	def test(self, image, point):
		cd, emd = self.sess.run([self.cd_loss, self.emd_loss], feed_dict = {self.image : image, self.pcl_gt[16384] : point})
		return cd, emd
	
	def predict(self, image):
		predicted_pointcloud = self.sess.run(self.pred, feed_dict = {self.image : image})
		return predicted_pointcloud

	def attention(self, x,is_training):
		#256 16
		x = self.image
		#256 16
		x = tf.layers.conv2d(x, filters=16, kernel_size=5, strides=2, padding='SAME')
		
		#128 32
		# conv, x -> [None, row, line, 32]
		x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding='SAME')
		x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding='SAME')
		x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=2, padding='SAME')

		# # max pooling, x -> [None, row, line, 32]
		# x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

		#64 64
		# attention module, x -> [None, row, line, 32]
		x = self.attention_module.f_prop(x, input_channels=64, scope="attention_module_1", is_training=is_training)

		# residual block, x-> [None, row, line, 64]
		x = self.residual_block.f_prop(x, input_channels=64, output_channels=128, scope="residual_block_1",
									   is_training=is_training)

		# max pooling, x -> [None, row, line, 64]
		x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

		#32 128
		x1 = x

		# attention module, x -> [None, row, line, 64]
		x = self.attention_module.f_prop(x, input_channels=128, scope="attention_module_2", is_training=is_training)

		# residual block, x-> [None, row, line, 128]
		x = self.residual_block.f_prop(x, input_channels=128, output_channels=256, scope="residual_block_2",
									   is_training=is_training)
		
		# max pooling, x -> [None, row/2, line/2, 128]
		x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


		#16 256
		x2 = x

		# attention module, x -> [None, row/2, line/2, 64]
		x = self.attention_module.f_prop(x, input_channels=256, scope="attention_module_3", is_training=is_training)

		# residual block, x-> [None, row/2, line/2, 256]
		x = self.residual_block.f_prop(x, input_channels=256, output_channels=512, scope="residual_block_3",
									   is_training=is_training)
		
		# max pooling, x -> [None, row/4, line/4, 256]
		x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

		#8 512
		x3 = x

		# residual block, x-> [None, row/4, line/4, 256]
		x = self.residual_block.f_prop(x, input_channels=512, output_channels=512, scope="residual_block_4",
									   is_training=is_training)

		# residual block, x-> [None, row/4, line/4, 256]
		x = self.residual_block.f_prop(x, input_channels=512, output_channels=512, scope="residual_block_5",
									   is_training=is_training)

		# residual block, x-> [None, row/4, line/4, 256]
		x = self.residual_block.f_prop(x, input_channels=512, output_channels=512, scope="residual_block_6",
									   is_training=is_training)

		# # 2048
		# x = tf.layers.dense(x, 2048)

		#4 512
		x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
		
		#8 256
		x=tflearn.layers.conv.conv_2d_transpose(x,256,[5,5],[8,8],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
		x3=tflearn.layers.conv.conv_2d(x3,256,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.nn.relu(tf.add(x,x3))
		x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		
		#16 128
		x=tflearn.layers.conv.conv_2d_transpose(x,128,[5,5],[16,16],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
		x2=tflearn.layers.conv.conv_2d(x2,128,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.nn.relu(tf.add(x,x2))
		x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		
		#32 64
		x=tflearn.layers.conv.conv_2d_transpose(x,64,[5,5],[32,32],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
		x1=tflearn.layers.conv.conv_2d(x1,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.nn.relu(tf.add(x,x1))
		x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')

		x=tflearn.layers.conv.conv_2d(x,3,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')

		x = tf.reshape(x,(tf.shape(x)[0],1024,3))

		# # average pooling
		# x = tf.nn.avg_pool(x, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')
		# x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))

		# # layer normalization
		# x = tf.contrib.layers.layer_norm(x, begin_norm_axis=-1)
		# # FC, softmax
		# x = tf.layers.dense(x, self.output_dim, activation=tf.nn.relu)
		return x

def pu_dense_dgcnn(x,is_training):

	upsample_factor = 4

	def dense_module(level, scale, point, *args):
		scale_idx = int(np.log(scale)/np.log(upsample_factor)) - 1
			
		batch_size = point.get_shape()[0].value
		num_input = point.get_shape()[1].value
		num_output = num_input * upsample_factor

		growth_rate=12
		comp=24
		dense_n=3
		k = 20
		use_bn=False
		use_ibn=False
		bn_decay=None
		# gt_original = gt
		
		with tf.variable_scope("level_%d" % (scale_idx+1), reuse=tf.AUTO_REUSE):

			# Feature extract
			# point feature
			point_feature = tf.expand_dims(point, axis=2) # (bs,num_input,3) --> (bs,num_input,1,3)
			
			# global feature (pointnet)
			global_feature = encoder_with_convs_and_symmetry(in_signal=point, n_filters=[32,64,64], 
								filter_sizes=[1],
								strides=[1],
								b_norm=True,
								verbose=False
								)	# (bs,64)
			global_feature = tf.tile(tf.expand_dims(global_feature, axis=1), [1,num_input,1]) # (bs,64) --> (bs,1,64) --> (bs,num_input,64)
			global_feature = tf.expand_dims(global_feature, axis=2) # (bs,num_input,64) --> (bs,num_input,1,64)

			# local feature (dgcnn)
			adj_matrix = pairwise_distance(point)
			nn_idx = knn(adj_matrix, k=k)
			edge_feature = get_edge_feature(point, nn_idx=nn_idx, k=k)

			net = conv2d(edge_feature, 32, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn1', bn_decay=bn_decay)
			net = tf.reduce_max(net, axis=-2, keep_dims=True)
			net1 = net

			adj_matrix = pairwise_distance(net)
			nn_idx = knn(adj_matrix, k=k)
			edge_feature = get_edge_feature(net, nn_idx=nn_idx, k=k)

			net = conv2d(edge_feature, 64, [1,1],
								padding='VALID', stride=[1,1],
								bn=True, is_training=is_training,
								scope='dgcnn2', bn_decay=bn_decay)
			net = tf.reduce_max(net, axis=-2, keep_dims=True)
			net2 = net

			adj_matrix = pairwise_distance(net)
			nn_idx = knn(adj_matrix, k=k)
			edge_feature = get_edge_feature(net, nn_idx=nn_idx, k=k)  
			
			net = conv2d(edge_feature, 128, [1,1],
								padding='VALID', stride=[1,1],
								bn=True, is_training=is_training,
								scope='dgcnn4', bn_decay=bn_decay)
			net = tf.reduce_max(net, axis=-2, keep_dims=True)
			net3 = net

			local_feature = conv2d(tf.concat([net1, net2, net3], axis=-1), 256, [1, 1], 
								padding='VALID', stride=[1,1],
								bn=True, is_training=is_training,
								scope='agg', bn_decay=bn_decay)
			


			# concat feature
			concat_feature = tf.concat([point_feature, global_feature, local_feature], axis=-1) # (bs,num_input,1,259)

			# Feature skip connect
			if scale_idx > 0:
				with tf.name_scope("skip_connection"):
					skip_feature = tf.get_collection("skip_feature_%d_%d" % (level-1, scale_idx - 1))
					skip_point = tf.get_collection("skip_point_%d_%d" % (level-1, scale_idx - 1))
					skip_feature = skip_feature.pop()	
					skip_point = skip_point.pop()
					# print 'skip_point', skip_point
					_dist, idx = knn_point(1, skip_point, point, sort=True, unique=True)
					# print 'skip_feature', skip_feature
					knn_feature = tf.gather_nd(tf.squeeze(skip_feature, axis=2), idx)
					# print 'concat_feature', concat_feature
					# print 'knn_feature', knn_feature

					concat_feature += 0.2*knn_feature 

			tf.add_to_collection("skip_point_%d_%d" % (level, scale_idx),
				tf.concat(tf.split(point, point.shape[0]//batch_size, axis=0), axis=1))
			tf.add_to_collection("skip_feature_%d_%d" % (level, scale_idx),
				tf.concat(tf.split(concat_feature, concat_feature.shape[0]//batch_size, axis=0), axis=1))

			# Feature expansion
			with tf.variable_scope('up_layer', reuse=tf.AUTO_REUSE):
				grid = tf.meshgrid(tf.linspace(-0.1,0.1,2), tf.linspace(-0.1,0.1,2))
				grid_feature = tf.tile(tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0), [batch_size, num_input, 1])

				new_feature = tf.reshape(tf.tile(concat_feature, [1,1,upsample_factor,1]), [batch_size, num_input*upsample_factor, -1]) # (bs, num_output, 259)
				new_feature = tf.concat([new_feature, grid_feature], axis=-1)

			output_pt = decoder_with_convs_only(new_feature, n_filters=[128,128,64,3], 
											filter_sizes=[1], 
											strides=[1],
											b_norm=True, 
											b_norm_finish=False, 
											verbose=False)
			
			print "pred shape", output_pt.shape
		return output_pt
	output_4k = dense_module(0,4,x)
	output_16k = dense_module(1,16,output_4k)

	return output_4k, output_16k
