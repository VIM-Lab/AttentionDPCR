import tensorflow as tf
from utils.datafetcher_sparse import DataFetcher
from model.sparse import Sparse
import time
import os
import sys

#python train.py model_name

if __name__ == '__main__':
	model_name = sys.argv[1]
	model_dir = os.path.join('result', model_name)
	if not os.path.isdir(model_dir):
		os.makedirs(model_dir)
	train_log = os.path.join(model_dir,'{}_train.log'.format(model_name, ))
	
	#Global variables setting
	epoch = 20
	batch_size = 32

	# Load data
	data = DataFetcher('train', batch_size = batch_size, epoch = epoch)
	data.setDaemon(True)
	data.start()
	train_number = data.iter
	
	#GPU settings 90% memory usage
	config = tf.ConfigProto() 
	config.gpu_options.per_process_gpu_memory_fraction = 0.9
	config.gpu_options.allow_growth = True

	with tf.Session(config = config) as sess: 
		model = Sparse(sess, 'train')
		sess.run(tf.global_variables_initializer())

		print('Training starts!')
		for e in range(epoch):
			print('---- Epoch {}/{} ----'.format(e + 1, epoch))
			model.saver.save(sess, os.path.join(model_dir, 'sparse_epoch_{}.ckpt'.format(e + 1)))
			for i in range(train_number):
				image, point = data.fetch()
				loss, cd = model.train(image, point)
				
				if i % 100 == 0 or i == train_number - 1:

					
					current_time = time.strftime("%m-%d %H:%M:%S", time.localtime())
					print('Epoch {} / {} iter {} / {} --- Loss:{} - CD:{} - time:{}'.format(e + 1, epoch, i + 1, train_number, loss, cd, current_time))
					with open(train_log, 'a+') as f:
						f.write('Epoch {} / {} iter {} / {} --- Loss:{} - CD:{} - time:{}\n'.format(e + 1, epoch, i + 1, train_number, loss, cd, current_time))
				
			model.saver.save(sess, os.path.join(model_dir, 'sparse_epoch_{}.ckpt'.format(e + 1)))
	data.shutdown()
	print('Training finished!')      

