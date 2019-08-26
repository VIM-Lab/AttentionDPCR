import tensorflow as tf
import numpy as np
from layer.tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample

def _variable_on_cpu(name, shape, initializer, use_fp16=False, trainable=True):
	"""Helper to create a Variable stored on CPU memory.
	Args:
		name: name of the variable
		shape: list of ints
		initializer: initializer for Variable
	Returns:
		Variable Tensor
	"""
	with tf.device('/cpu:0'):
		dtype = tf.float16 if use_fp16 else tf.float32
		var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
	return var

def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
	"""Helper to create an initialized Variable with weight decay.

	Note that the Variable is initialized with a truncated normal distribution.
	A weight decay is added only if one is specified.

	Args:
		name: name of the variable
		shape: list of ints
		stddev: standard deviation of a truncated Gaussian
		wd: add L2Loss weight decay multiplied by this float. If None, weight
				decay is not added for this Variable.
		use_xavier: bool, whether to use xavier initializer

	Returns:
		Variable Tensor
	"""
	if use_xavier:
		initializer = tf.contrib.layers.xavier_initializer()
	else:
		initializer = tf.truncated_normal_initializer(stddev=stddev)
	var = _variable_on_cpu(name, shape, initializer)
	if wd is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var


def conv1d(inputs,
					 num_output_channels,
					 kernel_size,
					 scope,
					 stride=1,
					 padding='SAME',
					 use_xavier=True,
					 stddev=1e-3,
					 weight_decay=0.0,
					 activation_fn=tf.nn.relu,
					 bn=False,
					 bn_decay=None,
					 is_training=None,
					 is_dist=False):
	""" 1D convolution with non-linear operation.

	Args:
		inputs: 3-D tensor variable BxLxC
		num_output_channels: int
		kernel_size: int
		scope: string
		stride: int
		padding: 'SAME' or 'VALID'
		use_xavier: bool, use xavier_initializer if true
		stddev: float, stddev for truncated_normal init
		weight_decay: float
		activation_fn: function
		bn: bool, whether to use batch norm
		bn_decay: float or float tensor variable in [0,1]
		is_training: bool Tensor variable

	Returns:
		Variable tensor
	"""
	with tf.variable_scope(scope) as sc:
		num_in_channels = inputs.get_shape()[-1].value
		kernel_shape = [kernel_size,
										num_in_channels, num_output_channels]
		kernel = _variable_with_weight_decay('weights',
																				 shape=kernel_shape,
																				 use_xavier=use_xavier,
																				 stddev=stddev,
																				 wd=weight_decay)
		outputs = tf.nn.conv1d(inputs, kernel,
													 stride=stride,
													 padding=padding)
		biases = _variable_on_cpu('biases', [num_output_channels],
															tf.constant_initializer(0.0))
		outputs = tf.nn.bias_add(outputs, biases)

		if bn:
			outputs = batch_norm_for_conv1d(outputs, is_training,
																			bn_decay=bn_decay, scope='bn', is_dist=is_dist)

		if activation_fn is not None:
			outputs = activation_fn(outputs)
		return outputs




def conv2d(inputs,
					 num_output_channels,
					 kernel_size,
					 scope,
					 stride=[1, 1],
					 padding='SAME',
					 use_xavier=True,
					 stddev=1e-3,
					 weight_decay=0.0,
					 activation_fn=tf.nn.relu,
					 bn=False,
					 bn_decay=None,
					 is_training=None,
					 is_dist=False):
	""" 2D convolution with non-linear operation.

	Args:
		inputs: 4-D tensor variable BxHxWxC
		num_output_channels: int
		kernel_size: a list of 2 ints
		scope: string
		stride: a list of 2 ints
		padding: 'SAME' or 'VALID'
		use_xavier: bool, use xavier_initializer if true
		stddev: float, stddev for truncated_normal init
		weight_decay: float
		activation_fn: function
		bn: bool, whether to use batch norm
		bn_decay: float or float tensor variable in [0,1]
		is_training: bool Tensor variable

	Returns:
		Variable tensor
	"""
	with tf.variable_scope(scope) as sc:
			kernel_h, kernel_w = kernel_size
			num_in_channels = inputs.get_shape()[-1].value
			kernel_shape = [kernel_h, kernel_w,
											num_in_channels, num_output_channels]
			kernel = _variable_with_weight_decay('weights',
																					 shape=kernel_shape,
																					 use_xavier=use_xavier,
																					 stddev=stddev,
																					 wd=weight_decay)
			stride_h, stride_w = stride
			outputs = tf.nn.conv2d(inputs, kernel,
														 [1, stride_h, stride_w, 1],
														 padding=padding)
			biases = _variable_on_cpu('biases', [num_output_channels],
																tf.constant_initializer(0.0))
			outputs = tf.nn.bias_add(outputs, biases)

			if bn:
				outputs = batch_norm_for_conv2d(outputs, is_training,
																				bn_decay=bn_decay, scope='bn', is_dist=is_dist)

			if activation_fn is not None:
				outputs = activation_fn(outputs)
			return outputs


def conv2d_transpose(inputs,
										 num_output_channels,
										 kernel_size,
										 scope,
										 stride=[1, 1],
										 padding='SAME',
										 use_xavier=True,
										 stddev=1e-3,
										 weight_decay=0.0,
										 activation_fn=tf.nn.relu,
										 bn=False,
										 bn_decay=None,
										 is_training=None,
										 is_dist=False):
	""" 2D convolution transpose with non-linear operation.

	Args:
		inputs: 4-D tensor variable BxHxWxC
		num_output_channels: int
		kernel_size: a list of 2 ints
		scope: string
		stride: a list of 2 ints
		padding: 'SAME' or 'VALID'
		use_xavier: bool, use xavier_initializer if true
		stddev: float, stddev for truncated_normal init
		weight_decay: float
		activation_fn: function
		bn: bool, whether to use batch norm
		bn_decay: float or float tensor variable in [0,1]
		is_training: bool Tensor variable

	Returns:
		Variable tensor

	Note: conv2d(conv2d_transpose(a, num_out, ksize, stride), a.shape[-1], ksize, stride) == a
	"""
	with tf.variable_scope(scope) as sc:
			kernel_h, kernel_w = kernel_size
			num_in_channels = inputs.get_shape()[-1].value
			kernel_shape = [kernel_h, kernel_w,
											num_output_channels, num_in_channels] # reversed to conv2d
			kernel = _variable_with_weight_decay('weights',
																					 shape=kernel_shape,
																					 use_xavier=use_xavier,
																					 stddev=stddev,
																					 wd=weight_decay)
			stride_h, stride_w = stride
			
			# from slim.convolution2d_transpose
			def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
					dim_size *= stride_size

					if padding == 'VALID' and dim_size is not None:
						dim_size += max(kernel_size - stride_size, 0)
					return dim_size

			# caculate output shape
			batch_size = inputs.get_shape()[0].value
			height = inputs.get_shape()[1].value
			width = inputs.get_shape()[2].value
			out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
			out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
			output_shape = [batch_size, out_height, out_width, num_output_channels]

			outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape,
														 [1, stride_h, stride_w, 1],
														 padding=padding)
			biases = _variable_on_cpu('biases', [num_output_channels],
																tf.constant_initializer(0.0))
			outputs = tf.nn.bias_add(outputs, biases)

			if bn:
				outputs = batch_norm_for_conv2d(outputs, is_training,
																				bn_decay=bn_decay, scope='bn', is_dist=is_dist)

			if activation_fn is not None:
				outputs = activation_fn(outputs)
			return outputs

	 

def conv3d(inputs,
					 num_output_channels,
					 kernel_size,
					 scope,
					 stride=[1, 1, 1],
					 padding='SAME',
					 use_xavier=True,
					 stddev=1e-3,
					 weight_decay=0.0,
					 activation_fn=tf.nn.relu,
					 bn=False,
					 bn_decay=None,
					 is_training=None,
					 is_dist=False):
	""" 3D convolution with non-linear operation.

	Args:
		inputs: 5-D tensor variable BxDxHxWxC
		num_output_channels: int
		kernel_size: a list of 3 ints
		scope: string
		stride: a list of 3 ints
		padding: 'SAME' or 'VALID'
		use_xavier: bool, use xavier_initializer if true
		stddev: float, stddev for truncated_normal init
		weight_decay: float
		activation_fn: function
		bn: bool, whether to use batch norm
		bn_decay: float or float tensor variable in [0,1]
		is_training: bool Tensor variable

	Returns:
		Variable tensor
	"""
	with tf.variable_scope(scope) as sc:
		kernel_d, kernel_h, kernel_w = kernel_size
		num_in_channels = inputs.get_shape()[-1].value
		kernel_shape = [kernel_d, kernel_h, kernel_w,
										num_in_channels, num_output_channels]
		kernel = _variable_with_weight_decay('weights',
																				 shape=kernel_shape,
																				 use_xavier=use_xavier,
																				 stddev=stddev,
																				 wd=weight_decay)
		stride_d, stride_h, stride_w = stride
		outputs = tf.nn.conv3d(inputs, kernel,
													 [1, stride_d, stride_h, stride_w, 1],
													 padding=padding)
		biases = _variable_on_cpu('biases', [num_output_channels],
															tf.constant_initializer(0.0))
		outputs = tf.nn.bias_add(outputs, biases)
		
		if bn:
			outputs = batch_norm_for_conv3d(outputs, is_training,
																			bn_decay=bn_decay, scope='bn', is_dist=is_dist)

		if activation_fn is not None:
			outputs = activation_fn(outputs)
		return outputs

def fully_connected(inputs,
										num_outputs,
										scope,
										use_xavier=True,
										stddev=1e-3,
										weight_decay=0.0,
										activation_fn=tf.nn.relu,
										bn=False,
										bn_decay=None,
										is_training=None,
										is_dist=False):
	""" Fully connected layer with non-linear operation.
	
	Args:
		inputs: 2-D tensor BxN
		num_outputs: int
	
	Returns:
		Variable tensor of size B x num_outputs.
	"""
	with tf.variable_scope(scope) as sc:
		num_input_units = inputs.get_shape()[-1].value
		weights = _variable_with_weight_decay('weights',
																					shape=[num_input_units, num_outputs],
																					use_xavier=use_xavier,
																					stddev=stddev,
																					wd=weight_decay)
		outputs = tf.matmul(inputs, weights)
		biases = _variable_on_cpu('biases', [num_outputs],
														 tf.constant_initializer(0.0))
		outputs = tf.nn.bias_add(outputs, biases)
		 
		if bn:
			outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn', is_dist=is_dist)

		if activation_fn is not None:
			outputs = activation_fn(outputs)
		return outputs


def max_pool2d(inputs,
							 kernel_size,
							 scope,
							 stride=[2, 2],
							 padding='VALID'):
	""" 2D max pooling.

	Args:
		inputs: 4-D tensor BxHxWxC
		kernel_size: a list of 2 ints
		stride: a list of 2 ints
	
	Returns:
		Variable tensor
	"""
	with tf.variable_scope(scope) as sc:
		kernel_h, kernel_w = kernel_size
		stride_h, stride_w = stride
		outputs = tf.nn.max_pool(inputs,
														 ksize=[1, kernel_h, kernel_w, 1],
														 strides=[1, stride_h, stride_w, 1],
														 padding=padding,
														 name=sc.name)
		return outputs

def avg_pool2d(inputs,
							 kernel_size,
							 scope,
							 stride=[2, 2],
							 padding='VALID'):
	""" 2D avg pooling.

	Args:
		inputs: 4-D tensor BxHxWxC
		kernel_size: a list of 2 ints
		stride: a list of 2 ints
	
	Returns:
		Variable tensor
	"""
	with tf.variable_scope(scope) as sc:
		kernel_h, kernel_w = kernel_size
		stride_h, stride_w = stride
		outputs = tf.nn.avg_pool(inputs,
														 ksize=[1, kernel_h, kernel_w, 1],
														 strides=[1, stride_h, stride_w, 1],
														 padding=padding,
														 name=sc.name)
		return outputs


def max_pool3d(inputs,
							 kernel_size,
							 scope,
							 stride=[2, 2, 2],
							 padding='VALID'):
	""" 3D max pooling.

	Args:
		inputs: 5-D tensor BxDxHxWxC
		kernel_size: a list of 3 ints
		stride: a list of 3 ints
	
	Returns:
		Variable tensor
	"""
	with tf.variable_scope(scope) as sc:
		kernel_d, kernel_h, kernel_w = kernel_size
		stride_d, stride_h, stride_w = stride
		outputs = tf.nn.max_pool3d(inputs,
															 ksize=[1, kernel_d, kernel_h, kernel_w, 1],
															 strides=[1, stride_d, stride_h, stride_w, 1],
															 padding=padding,
															 name=sc.name)
		return outputs

def avg_pool3d(inputs,
							 kernel_size,
							 scope,
							 stride=[2, 2, 2],
							 padding='VALID'):
	""" 3D avg pooling.

	Args:
		inputs: 5-D tensor BxDxHxWxC
		kernel_size: a list of 3 ints
		stride: a list of 3 ints
	
	Returns:
		Variable tensor
	"""
	with tf.variable_scope(scope) as sc:
		kernel_d, kernel_h, kernel_w = kernel_size
		stride_d, stride_h, stride_w = stride
		outputs = tf.nn.avg_pool3d(inputs,
															 ksize=[1, kernel_d, kernel_h, kernel_w, 1],
															 strides=[1, stride_d, stride_h, stride_w, 1],
															 padding=padding,
															 name=sc.name)
		return outputs





def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
	""" Batch normalization on convolutional maps and beyond...
	Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
	
	Args:
			inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
			is_training:   boolean tf.Varialbe, true indicates training phase
			scope:         string, variable scope
			moments_dims:  a list of ints, indicating dimensions for moments calculation
			bn_decay:      float or float tensor variable, controling moving average weight
	Return:
			normed:        batch-normalized maps
	"""
	with tf.variable_scope(scope) as sc:
		is_training = tf.cast(is_training, tf.bool)
		num_channels = inputs.get_shape()[-1].value
		beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
											 name='beta', trainable=True)
		gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
												name='gamma', trainable=True)
		batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
		decay = bn_decay if bn_decay is not None else 0.9
		ema = tf.train.ExponentialMovingAverage(decay=decay)
		# Operator that maintains moving averages of variables.
		ema_apply_op = tf.cond(is_training,
													 lambda: ema.apply([batch_mean, batch_var]),
													 lambda: tf.no_op())
		
		# Update moving average and return current batch's avg and var.
		def mean_var_with_update():
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)
		
		# ema.average returns the Variable holding the average of var.
		mean, var = tf.cond(is_training,
												mean_var_with_update,
												lambda: (ema.average(batch_mean), ema.average(batch_var)))
		normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
	return normed


def batch_norm_dist_template(inputs, is_training, scope, moments_dims, bn_decay):
	""" The batch normalization for distributed training.
	Args:
			inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
			is_training:   boolean tf.Varialbe, true indicates training phase
			scope:         string, variable scope
			moments_dims:  a list of ints, indicating dimensions for moments calculation
			bn_decay:      float or float tensor variable, controling moving average weight
	Return:
			normed:        batch-normalized maps
	"""
	with tf.variable_scope(scope) as sc:
		num_channels = inputs.get_shape()[-1].value
		beta = _variable_on_cpu('beta', [num_channels], initializer=tf.zeros_initializer())
		gamma = _variable_on_cpu('gamma', [num_channels], initializer=tf.ones_initializer())

		pop_mean = _variable_on_cpu('pop_mean', [num_channels], initializer=tf.zeros_initializer(), trainable=False)
		pop_var = _variable_on_cpu('pop_var', [num_channels], initializer=tf.ones_initializer(), trainable=False)

		def train_bn_op():
			batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
			decay = bn_decay if bn_decay is not None else 0.9
			train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay)) 
			train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
			with tf.control_dependencies([train_mean, train_var]):
				return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, 1e-3)

		def test_bn_op():
			return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, 1e-3)

		normed = tf.cond(is_training,
										 train_bn_op,
										 test_bn_op)
		return normed



def batch_norm_for_fc(inputs, is_training, bn_decay, scope, is_dist=False):
	""" Batch normalization on FC data.
	
	Args:
			inputs:      Tensor, 2D BxC input
			is_training: boolean tf.Varialbe, true indicates training phase
			bn_decay:    float or float tensor variable, controling moving average weight
			scope:       string, variable scope
			is_dist:     true indicating distributed training scheme
	Return:
			normed:      batch-normalized maps
	"""
	if is_dist:
		return batch_norm_dist_template(inputs, is_training, scope, [0,], bn_decay)
	else:
		return batch_norm_template(inputs, is_training, scope, [0,], bn_decay)


def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope, is_dist=False):
	""" Batch normalization on 1D convolutional maps.
	
	Args:
			inputs:      Tensor, 3D BLC input maps
			is_training: boolean tf.Varialbe, true indicates training phase
			bn_decay:    float or float tensor variable, controling moving average weight
			scope:       string, variable scope
			is_dist:     true indicating distributed training scheme
	Return:
			normed:      batch-normalized maps
	"""
	if is_dist:
		return batch_norm_dist_template(inputs, is_training, scope, [0,1], bn_decay)
	else:
		return batch_norm_template(inputs, is_training, scope, [0,1], bn_decay)



	
def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope, is_dist=False):
	""" Batch normalization on 2D convolutional maps.
	
	Args:
			inputs:      Tensor, 4D BHWC input maps
			is_training: boolean tf.Varialbe, true indicates training phase
			bn_decay:    float or float tensor variable, controling moving average weight
			scope:       string, variable scope
			is_dist:     true indicating distributed training scheme
	Return:
			normed:      batch-normalized maps
	"""
	if is_dist:
		return batch_norm_dist_template(inputs, is_training, scope, [0,1,2], bn_decay)
	else:
		return batch_norm_template(inputs, is_training, scope, [0,1,2], bn_decay)

def get_edge_feature(point_cloud, nn_idx, k=20):
	"""Construct edge feature for each point
	Args:
		point_cloud: (batch_size, num_points, 1, num_dims)
		nn_idx: (batch_size, num_points, k)
		k: int
	Returns:
		edge features: (batch_size, num_points, k, num_dims)
	"""
	og_batch_size = point_cloud.get_shape().as_list()[0]
	point_cloud = tf.squeeze(point_cloud)
	if og_batch_size == 1:
		point_cloud = tf.expand_dims(point_cloud, 0)

	point_cloud_central = point_cloud
	point_cloud_shape = point_cloud.get_shape()
	batch_size = point_cloud_shape[0].value
	num_points = point_cloud_shape[1].value
	num_dims = point_cloud_shape[2].value

	idx_ = tf.range(batch_size) * num_points
	idx_ = tf.reshape(idx_, [batch_size, 1, 1]) 

	point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
	point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)
	point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

	point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

	edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors-point_cloud_central], axis=-1)
	return edge_feature

def knn(adj_matrix, k=20):
	"""Get KNN based on the pairwise distance.
	Args:
		pairwise distance: (batch_size, num_points, num_points)
		k: int
	Returns:
		nearest neighbors: (batch_size, num_points, k)
	"""
	neg_adj = -adj_matrix
	_, nn_idx = tf.nn.top_k(neg_adj, k=k)
	return nn_idx

def pairwise_distance(point_cloud):
	"""Compute pairwise distance of a point cloud.
	Args:
		point_cloud: tensor (batch_size, num_points, num_dims)
	Returns:
		pairwise distance: (batch_size, num_points, num_points)
	"""
	og_batch_size = point_cloud.get_shape().as_list()[0]
	point_cloud = tf.squeeze(point_cloud)
	if og_batch_size == 1:
		point_cloud = tf.expand_dims(point_cloud, 0)

	point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])
	point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)
	point_cloud_inner = -2*point_cloud_inner
	point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True)
	point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
	return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose

def knn_point(k, points, queries, sort=True, unique=True):
		"""
		points: dataset points (N, P0, K)
		queries: query points (N, P, K)
		return indices is (N, P, K, 2) used for tf.gather_nd(points, indices)
		distances (N, P, K)
		"""
		with tf.name_scope("knn_point"):
				batch_size = tf.shape(queries)[0]
				point_num = tf.shape(queries)[1]

				D = batch_distance_matrix_general(queries, points)
				if unique:
						prepare_for_unique_top_k(D, points)
				distances, point_indices = tf.nn.top_k(-D, k=k, sorted=sort)  # (N, P, K)
				batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, point_num, k, 1))
				indices = tf.concat([batch_indices, tf.expand_dims(point_indices, axis=3)], axis=3)
				return -distances, indices

def prepare_for_unique_top_k(D, A):
		indices_duplicated = tf.py_func(find_duplicate_columns, [A], tf.int32)
		D += tf.reduce_max(D)*tf.cast(indices_duplicated, tf.float32)

def batch_distance_matrix_general(A, B):
		r_A = tf.reduce_sum(A * A, axis=2, keep_dims=True)
		r_B = tf.reduce_sum(B * B, axis=2, keep_dims=True)
		m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
		D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
		return D

def find_duplicate_columns(A):
		N = A.shape[0]
		P = A.shape[1]
		indices_duplicated = np.ones((N, 1, P), dtype=np.int32)
		for idx in range(N):
				_, indices = np.unique(A[idx], return_index=True, axis=0)
				indices_duplicated[idx, :, indices] = 0
		return indices_duplicated

def get_edge_feature_pu(point_cloud, k=20, idx=None):
		"""Construct edge feature for each point
		Args:
				point_cloud: (batch_size, num_points, 1, num_dims)
				nn_idx: (batch_size, num_points, k, 2)
				k: int
		Returns:
				edge features: (batch_size, num_points, k, num_dims)
		"""
		if idx is None:
				_, idx = knn_point(k+1, point_cloud, point_cloud, unique=True, sort=True)
				idx = idx[:, :, 1:, :]

		# [N, P, K, Dim]
		point_cloud_neighbors = tf.gather_nd(point_cloud, idx)
		point_cloud_central = tf.expand_dims(point_cloud, axis=-2)

		point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

		edge_feature = tf.concat(
				[point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)
		return edge_feature, idx

def extract_patch_for_next_level(batch_xyz, k, batch_features=None, gt_xyz=None, gt_k=None, is_training=True):
		"""
		:param batch_xyz [B, P, 3]
		"""
		batch_size, num_point, _ = batch_xyz.shape.as_list()
		with tf.name_scope("extract_input"):
				if is_training:
						# B, 1, 3
						idx = tf.random_uniform([batch_size, 1], minval=0, maxval=num_point, dtype=tf.int32)
						# idx = tf.constant(250, shape=[batch_size, 1], dtype=tf.int32)
						batch_seed_point = gather_point(batch_xyz, idx)
						patch_num = 1
				else:
						assert(batch_size == 1)
						# remove residual, (B P 1) and (B, P, 1, 2)
						closest_d, _ = knn_point(2, batch_xyz, batch_xyz, unique=False)
						closest_d = closest_d[:, :, 1:]
						# (B, P)
						mask = tf.squeeze(closest_d < 5*(tf.reduce_mean(closest_d, axis=1, keep_dims=True)), axis=-1)
						# filter (B, P', 3)
						batch_xyz = tf.expand_dims(tf.boolean_mask(batch_xyz, mask), axis=0)
						# batch_xyz = tf.Print(batch_xyz, [tf.shape(batch_xyz)])
						# B, M, 3
						# batch_seed_point = batch_xyz[:, -1:, :]
						patch_num = 1
						# patch_num = int(num_point / k * 5)
						# print 'patch_num',patch_num
						# idx = tf.random_uniform([batch_size, patch_num], minval=0, maxval=num_point, dtype=tf.int32)
						idx = tf.squeeze(farthest_point_sample(patch_num, batch_xyz), axis=0)
						# idx = tf.random_uniform([patch_num], minval=0, maxval=tf.shape(batch_xyz)[1], dtype=tf.int32)
						# B, P, 3 -> B, k, 3 (idx B, k, 1)
						# idx = tf.Print(idx, [idx], message="idx")
						batch_seed_point = tf.gather(batch_xyz, idx, axis=1)
						k = tf.minimum(k, tf.shape(batch_xyz)[1])
						# batch_seed_point = gather_point(batch_xyz, idx)
				# B, M, k, 2
				_, new_patch_idx = knn_point(k, batch_xyz, batch_seed_point, unique=False)
				# B, M, k, 3
				batch_xyz = tf.gather_nd(batch_xyz, new_patch_idx)
				# MB, k, 3
				batch_xyz = tf.concat(tf.unstack(batch_xyz, axis=1), axis=0)
				
		if batch_features is not None:
				with tf.name_scope("extract_feature"):
						batch_features = tf.gather_nd(batch_features, new_patch_idx)
						batch_features = tf.concat(tf.unstack(batch_features, axis=1), axis=0)
		if is_training and (gt_xyz is not None and gt_k is not None):
				with tf.name_scope("extract_gt"):
						_, new_patch_idx = knn_point(gt_k, gt_xyz, batch_seed_point, unique=False)
						gt_xyz = tf.gather_nd(gt_xyz, new_patch_idx)
						gt_xyz = tf.concat(tf.unstack(gt_xyz, axis=1), axis=0)
		else:
				gt_xyz = None

		return batch_xyz, batch_features, gt_xyz

def normalize_point_cloud(pc):
		"""
		pc [N, P, 3]
		"""
		centroid = tf.reduce_mean(pc, axis=1, keep_dims=True)
		pc = pc - centroid
		furthest_distance = tf.reduce_max(
				tf.sqrt(tf.reduce_sum(pc ** 2, axis=-1, keep_dims=True)), axis=1, keep_dims=True)
		pc = pc / furthest_distance
		return pc, centroid, furthest_distance
