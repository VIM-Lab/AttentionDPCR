import tensorflow as tf
from utils.datafetcher_finetune import DataFetcher
from model.dense import Finetune
import time
import os
import sys

#python train.py model_name

def load_ckpt(model):
	model_dir = model_dir = os.path.join('result', 'sparse')
	ckpt = tf.train.get_checkpoint_state(model_dir)
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print('Reloading attention_baseline parameters')
		model.saver_sparse.restore(sess, ckpt.model_checkpoint_path)
	else:
		raise ValueError('No such file:[{}]'.format(model_dir))
	model_dir = model_dir = os.path.join('result', 'dense')
	ckpt = tf.train.get_checkpoint_state(model_dir)
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print('Reloading my_pu_comlex parameters')
		model.saver_dense.restore(sess, ckpt.model_checkpoint_path)
	else:
		raise ValueError('No such file:[{}]'.format(model_dir))


if __name__ == '__main__':
	model_name = sys.argv[1]
	model_dir = os.path.join('result', model_name)
	if not os.path.isdir(model_dir):
		os.makedirs(model_dir)
	train_log = os.path.join(model_dir,'{}_train.log'.format(model_name, ))
	
	#Global variables setting
	epoch = 40
	batch_size = 10

	# Load data
	data = DataFetcher('train', epoch = epoch, batch_size = batch_size)
	data.setDaemon(True)
	data.start()
	train_number = data.iter
	
	#GPU settings 90% memory usage
	config = tf.ConfigProto() 
	config.gpu_options.per_process_gpu_memory_fraction = 0.9
	config.gpu_options.allow_growth = True

	with tf.Session(config = config) as sess:    
		model = Finetune(sess, 'train', batch_size)
		sess.run(tf.global_variables_initializer())
		start_epoch = 0
		ckpt = tf.train.get_checkpoint_state(model_dir)
		if ckpt is not None:
			print ( 'loading ' + ckpt.model_checkpoint_path + ' ...')
			model.saver.restore(sess, ckpt.model_checkpoint_path)
			start_epoch = int(ckpt.model_checkpoint_path.split('.')[0].split('_')[-1])
		print('Training starts!')
		for e in range(start_epoch, epoch):
			model.saver.save(sess, os.path.join(model_dir, 'finetune_epoch_{}.ckpt'.format(e + 1)))
			if e == 0:
				print('restoring previous epoches')
				load_ckpt(model)
			print('---- Epoch {}/{} ----'.format(e + 1, epoch))

			for i in range(train_number):
				image, point1, point2 = data.fetch()
				loss, cd = model.train(image, point1, point2)
				
				if i % 100 == 0 or i == train_number - 1:
					current_time = time.strftime("%m-%d %H:%M:%S", time.localtime())
					print('Epoch {} / {} iter {} / {} --- Loss:{} - CD:{} - time:{}'.format(e + 1, epoch, i + 1, train_number, loss, cd, current_time))
					with open(train_log, 'a+') as f:
						f.write('Epoch {} / {} iter {} / {} --- Loss:{} - CD:{} - time:{}\n'.format(e + 1, epoch, i + 1, train_number, loss, cd, current_time))
					
			model.saver.save(sess, os.path.join(model_dir, 'finetune_epoch_{}.ckpt'.format(e + 1)))
	data.shutdown()
	print('Training finished!')      

