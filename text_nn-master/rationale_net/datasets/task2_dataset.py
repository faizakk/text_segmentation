import gzip
import re
import tqdm
from rationale_net.utils.embedding import get_indices_tensor
from rationale_net.datasets.factory import RegisterDataset
from rationale_net.datasets.abstract_dataset import AbstractDataset
from sklearn.datasets import fetch_20newsgroups
import random
import pandas as pd
random.seed(0)


#SMALL_TRAIN_SIZE = 800
#CATEGORIES =....
all_data= pd.read_csv('/h/faizakk/text_nn-master/rationale_net/datasets/ROC_data/task2_data.csv', error_bad_lines=False)


def preprocess_data(data):
	processed_data = []

	label_list,uniques = pd.factorize(list(data.construct))
	print('uniques',uniques) 	
	for indx, sample in enumerate(data['abstract']):
		#print('testing sample', sample)
		text, label_name = sample, data['construct'][indx]
		label = label_list[indx]
		#print('testing sample', label_name,label)
		text = re.sub('\W+', ' ', text).lower().strip()
		processed_data.append( (text, label, label_name) )
		#CAT = data.Com_Diagnoses.unique()
		#print('testing the tuple',(processed_data[indx]))
	return processed_data



def preprocess_data_old(data):
    processed_data = []
    for indx, sample in enumerate(data['data']):
        text, label = sample[indx], data['target'][indx]
        label_name = data['target_names'][label][indx]
        text = re.sub('\W+', ' ', text).lower().strip()
        processed_data.append( (text, label, label_name) )
    return processed_data

@RegisterDataset('task2')
class AutoScribeDataset(AbstractDataset):

    def __init__(self, args, word_to_indx, name, max_length=2000): #20000): #20000): #20000): #20000): #20000): #20000): #20000): #20000): #20000): #20000): #20000): #20000): #20000): #20000): #20000): #20000): #20000): #20000): #20000): #41288):
        self.args = args
        self.args.num_class = 8
        self.name = name
        self.dataset = []
        self.word_to_indx  = word_to_indx
        self.max_length = max_length
        self.class_balance = {}

        if name in ['train', 'dev']:
            data = preprocess_data(all_data);print('all_data',data)
            random.shuffle(data)
            num_train = int(len(data)*.8);print('num_train',num_train)
            if name == 'train':
                data = data[:num_train]; print(data)
            else:
                data = data[num_train:]
        else:
            data = preprocess_data(all_data)

        for indx, _sample in tqdm.tqdm(enumerate(data)):
            sample = self.processLine(_sample); #print('sample',sample['y'],self.class_balance)
	
            if not sample['y'] in self.class_balance:
                self.class_balance[ sample['y'] ] = 0
            self.class_balance[ sample['y'] ] += 1
            self.dataset.append(sample)

        print ("Class balance", self.class_balance)

        if args.class_balance:
            raise NotImplementedError("NewsGroup dataset doesn't support balanced sampling")
        if args.objective == 'mse':
            raise NotImplementedError("News Group does not support Regression objective")

    ## Convert one line from beer dataset to {Text, Tensor, Labels}
    def processLine(self, row):
        text, label, label_name = row
        text = " ".join(text.split()[:self.max_length])
        x =  get_indices_tensor(text.split(), self.word_to_indx, self.max_length)
        sample = {'text':text,'x':x, 'y':label, 'y_name': label_name}
        return sample
