from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))
import argparse

import rationale_net.datasets.factory as dataset_factory
import rationale_net.utils.embedding as embedding
import rationale_net.utils.model as model_factory
import rationale_net.utils.generic as generic
import rationale_net.learn.train as train
import os
import torch
import datetime
import pickle
import pdb


if __name__ == '__main__':
	# update args and print
	args = generic.parse_args()

	embeddings, word_to_indx = embedding.get_embedding_tensor(args)

	train_data, dev_data, test_data = dataset_factory.get_dataset(args, word_to_indx)

	results_path_stem = args.results_path.split('/')[-1].split('.')[0]
	args.model_path = '{}.pt'.format(os.path.join(args.save_dir, results_path_stem))
	#print('test',args)
	# model
	gen, model = model_factory.get_model(args, embeddings, train_data)

	#print("CHECKING TRAINING DATA",train_data)
	print()
	# train
	if args.train :
		epoch_stats, model, gen = train.train_model(train_data, dev_data, model, gen, args)
		args.epoch_stats = epoch_stats
		save_path = args.results_path
		print("Save train/dev results to", save_path)
		args_dict = vars(args)
		pickle.dump(args_dict, open(save_path,'wb') )


	# test
	if args.test :
		test_stats = train.test_model(test_data, model, gen, args)
		args.test_stats = test_stats
		args.train_data = train_data
		args.test_data = test_data

		save_path = args.results_path
		print("Save test results to", save_path)
		args_dict = vars(args)
		pickle.dump(args_dict, open(save_path,'wb') )
		#print('arg_dict',args_dict,test_stats)
print('I am done')
