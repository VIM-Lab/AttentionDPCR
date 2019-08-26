import tensorflow as tf
import tflearn
from tensorflow.python.keras.layers import UpSampling2D

def attention_module(input, input_channels, scope="attention_module", is_training=True):
	p = 1
	t = 2
	r = 1

	with tf.variable_scope(scope):
		# residual blocks(TODO: change this function)
		with tf.variable_scope("first_residual_blocks"):
			for i in range(p):
				input = residual_block(input, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

		with tf.variable_scope("trunk_branch"):
			output_trunk = input
			for i in range(t):
				output_trunk = residual_block(output_trunk, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

		with tf.variable_scope("soft_mask_branch"):

			with tf.variable_scope("down_sampling_1"):
				# max pooling
				filter_ = [1, 2, 2, 1]
				output_soft_mask = tf.nn.max_pool(input, ksize=filter_, strides=filter_, padding='SAME')

				for i in range(r):
					output_soft_mask = residual_block(output_soft_mask, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

			with tf.variable_scope("skip_connection"):
				# TODO(define new blocks)
				output_skip_connection = residual_block(output_soft_mask, input_channels, is_training=is_training)


			with tf.variable_scope("down_sampling_2"):
				# max pooling
				filter_ = [1, 2, 2, 1]
				output_soft_mask = tf.nn.max_pool(output_soft_mask, ksize=filter_, strides=filter_, padding='SAME')

				for i in range(r):
					output_soft_mask = residual_block(output_soft_mask, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

			with tf.variable_scope("up_sampling_1"):
				for i in range(r):
					output_soft_mask = residual_block(output_soft_mask, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

				# interpolation
				# output_soft_mask = tf.image.resize_nearest_neighbor(output_soft_mask, (2*tf.shape(output_soft_mask)[1],2*tf.shape(output_soft_mask)[2]))
				output_soft_mask = UpSampling2D([2, 2])(output_soft_mask)

			# add skip connection
			output_soft_mask += output_skip_connection

			with tf.variable_scope("up_sampling_2"):
				for i in range(r):
					output_soft_mask = residual_block(output_soft_mask, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

				# interpolation
				# output_soft_mask = tf.image.resize_nearest_neighbor(output_soft_mask, (2*tf.shape(output_soft_mask)[1],2*tf.shape(output_soft_mask)[2]))
				output_soft_mask = UpSampling2D([2, 2])(output_soft_mask)


			with tf.variable_scope("output"):
				output_soft_mask = tf.layers.conv2d(output_soft_mask, filters=input_channels, kernel_size=1)
				output_soft_mask = tf.layers.conv2d(output_soft_mask, filters=input_channels, kernel_size=1)

				# sigmoid
				output_soft_mask = tf.nn.sigmoid(output_soft_mask)

		with tf.variable_scope("attention"):
			output = (1 + output_soft_mask) * output_trunk

		with tf.variable_scope("last_residual_blocks"):
			for i in range(p):
				output = residual_block(output, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

		return output

def attention_module_for_vis(input, input_channels, scope="attention_module", is_training=True):
	p = 1
	t = 2
	r = 1
	before = input
	with tf.variable_scope(scope):
		# residual blocks(TODO: change this function)
		with tf.variable_scope("first_residual_blocks"):
			for i in range(p):
				input = residual_block(input, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

		with tf.variable_scope("trunk_branch"):
			output_trunk = input
			for i in range(t):
				output_trunk = residual_block(output_trunk, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

		with tf.variable_scope("soft_mask_branch"):

			with tf.variable_scope("down_sampling_1"):
				# max pooling
				filter_ = [1, 2, 2, 1]
				output_soft_mask = tf.nn.max_pool(input, ksize=filter_, strides=filter_, padding='SAME')

				for i in range(r):
					output_soft_mask = residual_block(output_soft_mask, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

			with tf.variable_scope("skip_connection"):
				# TODO(define new blocks)
				output_skip_connection = residual_block(output_soft_mask, input_channels, is_training=is_training)


			with tf.variable_scope("down_sampling_2"):
				# max pooling
				filter_ = [1, 2, 2, 1]
				output_soft_mask = tf.nn.max_pool(output_soft_mask, ksize=filter_, strides=filter_, padding='SAME')

				for i in range(r):
					output_soft_mask = residual_block(output_soft_mask, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

			with tf.variable_scope("up_sampling_1"):
				for i in range(r):
					output_soft_mask = residual_block(output_soft_mask, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

				# interpolation
				output_soft_mask = tf.image.resize_nearest_neighbor(output_soft_mask, (2*tf.shape(output_soft_mask)[1],2*tf.shape(output_soft_mask)[2]))
				# output_soft_mask = UpSampling2D([2, 2])(output_soft_mask)

			# add skip connection
			output_soft_mask += output_skip_connection

			with tf.variable_scope("up_sampling_2"):
				for i in range(r):
					output_soft_mask = residual_block(output_soft_mask, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

				# interpolation
				output_soft_mask = tf.image.resize_nearest_neighbor(output_soft_mask, (2*tf.shape(output_soft_mask)[1],2*tf.shape(output_soft_mask)[2]))
				# output_soft_mask = UpSampling2D([2, 2])(output_soft_mask)


			with tf.variable_scope("output"):
				output_soft_mask = tf.layers.conv2d(output_soft_mask, filters=input_channels, kernel_size=1)
				output_soft_mask = tf.layers.conv2d(output_soft_mask, filters=input_channels, kernel_size=1)

				# sigmoid
				output_soft_mask = tf.nn.sigmoid(output_soft_mask)

		with tf.variable_scope("attention"):
			output = (1 + output_soft_mask) * output_trunk

		with tf.variable_scope("last_residual_blocks"):
			for i in range(p):
				output = residual_block(output, input_channels, scope="num_blocks_{}".format(i), is_training=is_training)

		return output, output_soft_mask, before

def residual_block(_input, input_channels, output_channels=None, scope="residual_block", is_training=True):
	if output_channels is None:
		output_channels = input_channels

	with tf.variable_scope(scope):
		# batch normalization & ReLU TODO(this function should be updated when the TF version changes)
		x = batch_norm(_input, input_channels, is_training)

		x = tf.layers.conv2d(x, filters=output_channels, kernel_size=1, padding='SAME', name="conv1")

		# batch normalization & ReLU TODO(this function should be updated when the TF version changes)
		x = batch_norm(x, output_channels, is_training)

		x = tf.layers.conv2d(x, filters=output_channels, kernel_size=3,
								strides=1, padding='SAME', name="conv2")

		# update input
		if input_channels != output_channels:
			_input = tf.layers.conv2d(_input, filters=output_channels, kernel_size=1, strides=1)

		output = x + _input

		return output

def batch_norm(x, n_out, is_training=True):
	"""
	Batch normalization on convolutional maps.
	Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
	Args:
		x:           Tensor, 4D BHWD input maps
		n_out:       integer, depth of input maps
		is_training: boolean tf.Varialbe, true indicates training phase
		scope:       string, variable scope
	Return:
		normed:      batch-normalized maps
	"""
	with tf.variable_scope('batch_norm'):
		beta = tf.Variable(lambda: tf.constant(0.0, shape=[n_out]),
							name='beta', trainable=True)
		gamma = tf.Variable(lambda: tf.constant(1.0, shape=[n_out]),
							name='gamma', trainable=True)
		batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
		ema = tf.train.ExponentialMovingAverage(decay=0.5)

		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		mean, var = tf.cond(tf.cast(is_training, tf.bool),
							mean_var_with_update,
							lambda: (ema.average(batch_mean), ema.average(batch_var)))
		normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
	return tf.nn.relu(normed)

def conv2d(layer, filters, filter_size, strides, activation='linear', weight_decay=1e-5, regularizer='L2', scope='conv'):
	with tf.variable_scope(scope):
		layer = tflearn.layers.conv.conv_2d(layer, filters, filter_size, strides, activation=activation, weight_decay=weight_decay, regularizer=regularizer)
	return layer

def deconv2d(layer, filters, filter_size, output_shape, strides=2, activation='linear', weight_decay=1e-5, regularizer='L2', scope='deconv'):
	with tf.variable_scope(scope):
		layer = tflearn.layers.conv.conv_2d_transpose(layer, filters, filter_size, output_shape, strides=strides, activation=activation, weight_decay=weight_decay, regularizer=regularizer)
	return layer

