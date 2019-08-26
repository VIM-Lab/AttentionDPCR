import tensorflow as tf
import tflearn
from layer.basic_layers import ResidualBlock
from layer.attention_module import AttentionModule
from metric.tf_nndistance import nn_distance
from metric.tf_approxmatch import approx_match, match_cost
import numpy as np

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

class Sparse(object):
	def __init__(self, sess, mode):
		self.sess = sess
		self.mode = mode

		self.attention_module = AttentionModule()
		self.residual_block = ResidualBlock()
		#initialize input and output
		self.image = tf.placeholder(tf.float32, shape = (None, 256, 256, 3), name = 'image')
		self.point = tf.placeholder(tf.float32, shape = (None, 16384, 3), name = 'point')
		# self.point = tf.placeholder(tf.float32, shape = (None, 1024, 3), name = 'point')
		self.gt = None
		self.pred = None

		#initialize before build graph
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
			self.build_optimizer()
		elif self.mode == 'test':
			self.build_graph(is_training=False)
			self.build_loss_calculater()
		elif self.mode == 'evaluate':
			self.build_graph(is_training=False)
			self.build_loss()
		else:
			self.build_graph(is_training=False)

	def build_optimizer(self):
		print('Building chamfer distance loss optimizer...')

		self.optimizer = tf.train.AdamOptimizer(3e-5)
		dist1,idx1,dist2,idx2 = nn_distance(self.gt, self.pred)
		loss_nodecay = (tf.reduce_mean(dist1) + 0.55 * tf.reduce_mean(dist2)) * 10000
		self.train_loss = loss_nodecay + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) * 0.1
		self.cd_loss = tf.reduce_mean(dist1) + tf.reduce_mean(dist2)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.opt_op = self.optimizer.minimize(self.train_loss)

	def build_loss_calculater(self):
		if self.mode == 'test':
			self.gt, self.pred = scale(self.gt, self.pred)
		#cd
		dist1,idx1,dist2,idx2 = nn_distance(self.gt, self.pred)
		self.cd_loss = tf.reduce_mean(dist1) + tf.reduce_mean(dist2)
		#emd
		match = approx_match(self.pred, self.gt)
		# self.emd_loss = tf.reduce_mean(match_cost(self.gt, self.pred, match)) / float(tf.shape(self.pred)[1])
		self.emd_loss = tf.reduce_mean(match_cost(self.pred, self.gt, match))
	
	def build_loss(self):
		
		self.scaled_gt, self.scaled_pred = scale(self.gt, self.pred)

		#cd
		dist1,idx1,dist2,idx2 = nn_distance(self.scaled_gt, self.scaled_pred)
		self.cd_loss = tf.reduce_mean(dist1) + tf.reduce_mean(dist2)
		#emd
		match = approx_match(self.scaled_pred, self.scaled_gt)
		# self.emd_loss = tf.reduce_mean(match_cost(self.gt, self.pred, match)) / float(tf.shape(self.pred)[1])
		self.emd_loss = tf.reduce_mean(match_cost(self.scaled_pred, self.scaled_gt, match))


	def train(self, image, point):
		_, train_loss, cd_loss = self.sess.run([self.opt_op, self.train_loss, self.cd_loss], feed_dict = {self.image : image, self.point : point})
		return train_loss, cd_loss
	
	def test(self, image, point):
		cd, emd = self.sess.run([self.cd_loss, self.emd_loss], feed_dict = {self.image : image, self.point : point})
		return cd, emd
	
	def predict(self, image):
		predicted_pointcloud = self.sess.run(self.pred, feed_dict = {self.image : image})
		return predicted_pointcloud

	def evaluate(self, image, point):		
		cd, emd, pred = self.sess.run([self.cd_loss, self.emd_loss, self.scaled_pred], feed_dict={self.image : image, self.point : point})
		return cd, emd, pred

	def build_graph(self, is_training = True):
		print('Building residual attention network...')
		# tflearn.is_training(is_training, session=self.sess)
		x = self.image
		with tf.variable_scope('sparse'):
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

		self.gt = self.point
		self.pred = x
