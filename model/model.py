import tensorflow as tf
import tflearn
from metric.tf_nndistance import nn_distance
from metric.tf_approxmatch import approx_match, match_cost

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

#build model
class Model(object):
	def __init__(self, sess, mode):
		self.sess = sess
		self.mode = mode

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

	def build_graph(self, is_training = True):
		raise NotImplementedError()
	
	def build_optimizer(self):
		print('Building chamfer distance loss optimizer...')

		self.optimizer = tf.train.AdamOptimizer(3e-5)
		dist1,idx1,dist2,idx2 = nn_distance(self.gt, self.pred)
		loss_nodecay = (tf.reduce_mean(dist1) + 0.55 * tf.reduce_mean(dist2)) * 10000
		self.train_loss = loss_nodecay + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) * 0.1
		self.cd_loss = tf.reduce_mean(dist1) + tf.reduce_mean(dist2)
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
		# self.ph_pred = tf.placeholder(tf.float32, shape = (None, 1024, 3), name = 'ph_pred')
		# self.ph_gt = tf.placeholder(tf.float32, shape = (None, 16384, 3), name = 'ph_gt')
		# self.ph_pred2 = tf.placeholder(tf.float32, shape = (None, 1024, 3), name = 'ph_pred2')
		# self.ph_gt2 = tf.placeholder(tf.float32, shape = (None, 1024, 3), name = 'ph_gt2')
		
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
		# scaled_gt, scaled_pred = self.sess.run([self.scaled_gt, self.scaled_pred], feed_dict={self.image : image, self.point : point})
		# temp = scaled_pred
		# _pr_scaled_icp = []
		# for i in range(1):
		# 	T, _, _ = icp(scaled_gt[i], scaled_pred[i], tolerance=1e-10, max_iterations=1000)
		# 	_pr_scaled_icp.append(np.dot(scaled_pred[i], T[:3,:3]) - T[:3, 3])
		# 	# _pr_scaled_icp.append(np.dot(T, scaled_pred[i].T).T)
		# scaled_pred = np.array(_pr_scaled_icp).astype('float32')

		# _gt_scaled_icp = []
		# for i in range(1):
		# 	T, _, _ = icp(scaled_pred[i], scaled_gt[i], tolerance=1e-10, max_iterations=1000)
		# 	_gt_scaled_icp.append(np.dot(scaled_gt[i], T[:3,:3]) - T[:3, 3])
		# 	# _pr_scaled_icp.append(np.dot(T, scaled_pred[i].T).T)
		# scaled_gt = np.array(_gt_scaled_icp).astype('float32')
		
		cd, emd, pred = self.sess.run([self.cd_loss, self.emd_loss, self.scaled_pred], feed_dict={self.image : image, self.point : point})
		return cd, emd, pred

class PSGN(Model):
	def __init__(self, sess, mode):
		super(PSGN, self).__init__(sess, mode)

	def build_graph(self, is_training=True):
		print('Building PSGN network...')
		tflearn.is_training(is_training, session=self.sess)

		#network structure
		#input
		x = self.image

#192 256
		x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x0=x
		x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#96 128
		x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x1=x
		x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#48 64
		x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x2=x
		x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#24 32
		x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x3=x
		x=tflearn.layers.conv.conv_2d(x,256,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#12 16
		x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x4=x
		x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#6 8
		x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x5=x
		x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#3 4
		x_additional=tflearn.layers.core.fully_connected(x,2048,activation='relu',weight_decay=1e-3,regularizer='L2')
		x=tflearn.layers.conv.conv_2d_transpose(x,256,[5,5],[6,8],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#6 8
		x5=tflearn.layers.conv.conv_2d(x5,256,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.nn.relu(tf.add(x,x5))
		x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x5=x  
		x=tflearn.layers.conv.conv_2d_transpose(x,128,[5,5],[12,16],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#12 16
		x4=tflearn.layers.conv.conv_2d(x4,128,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.nn.relu(tf.add(x,x4))
		x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x4=x
		x=tflearn.layers.conv.conv_2d_transpose(x,64,[5,5],[24,32],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#24 32
		x3=tflearn.layers.conv.conv_2d(x3,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.nn.relu(tf.add(x,x3))
		x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x3=x
		x=tflearn.layers.conv.conv_2d_transpose(x,32,[5,5],[48,64],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#48 64
		x2=tflearn.layers.conv.conv_2d(x2,32,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.nn.relu(tf.add(x,x2))
		x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x2=x
		x=tflearn.layers.conv.conv_2d_transpose(x,16,[5,5],[96,128],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#96 128
		x1=tflearn.layers.conv.conv_2d(x1,16,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.nn.relu(tf.add(x,x1))
		x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#48 64
		x2=tflearn.layers.conv.conv_2d(x2,32,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.nn.relu(tf.add(x,x2))
		x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x2=x
		x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#24 32
		x3=tflearn.layers.conv.conv_2d(x3,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.nn.relu(tf.add(x,x3))
		x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x3=x
		x=tflearn.layers.conv.conv_2d(x,128,(5,5),strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#12 16
		x4=tflearn.layers.conv.conv_2d(x4,128,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.nn.relu(tf.add(x,x4))
		x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x4=x
		x=tflearn.layers.conv.conv_2d(x,256,(5,5),strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#6 8
		x5=tflearn.layers.conv.conv_2d(x5,256,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.nn.relu(tf.add(x,x5))
		x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x5=x
		x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
#3 4
		x_additional=tflearn.layers.core.fully_connected(x_additional,2048,activation='linear',weight_decay=1e-4,regularizer='L2')
		x_additional=tf.nn.relu(tf.add(x_additional,tflearn.layers.core.fully_connected(x,2048,activation='linear',weight_decay=1e-3,regularizer='L2')))
		x=tflearn.layers.conv.conv_2d_transpose(x,256,[5,5],[6,8],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#6 8
		x5=tflearn.layers.conv.conv_2d(x5,256,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.nn.relu(tf.add(x,x5))
		x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x5=x  
		x=tflearn.layers.conv.conv_2d_transpose(x,128,[5,5],[12,16],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#12 16
		x4=tflearn.layers.conv.conv_2d(x4,128,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.nn.relu(tf.add(x,x4))
		x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x4=x
		x=tflearn.layers.conv.conv_2d_transpose(x,64,[5,5],[24,32],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
#24 32
		x3=tflearn.layers.conv.conv_2d(x3,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.nn.relu(tf.add(x,x3))
		x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')

		x_additional=tflearn.layers.core.fully_connected(x_additional,1024,activation='relu',weight_decay=1e-3,regularizer='L2')
		x_additional=tflearn.layers.core.fully_connected(x_additional,256*3,activation='linear',weight_decay=1e-3,regularizer='L2')
		x_additional=tf.reshape(x_additional,(tf.shape(x)[0],256,3))
		x=tflearn.layers.conv.conv_2d(x,3,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		x=tf.reshape(x,(tf.shape(x)[0],32*24,3))
		x=tf.concat([x_additional,x],1)
		x=tf.reshape(x,(tf.shape(x)[0],1024,3))

		#output
		self.gt = self.point
		self.pred = x
