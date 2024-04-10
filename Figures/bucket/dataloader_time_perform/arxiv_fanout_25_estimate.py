import sys
sys.path.insert(0,'..')
sys.path.insert(0,'..')
sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/utils/')
sys.path.insert(0,'/home/cc/Betty_baseline//pytorch/bucketing/')
sys.path.insert(0,'/home/cc/Betty_baseline//pytorch/models/')
import dgl
from dgl.data.utils import save_graphs
import numpy as np
from statistics import mean
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from bucketing_dataloader import generate_dataloader_bucket_block, dataloader_gen_range
from bucketing_dataloader import dataloader_gen_bucketing
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm

import random
from graphsage_model_wo_mem import GraphSAGE
import dgl.function as fn
from load_graph import load_reddit, inductive_split, load_ogb, load_cora, load_karate, prepare_data, load_pubmed

from load_graph import load_ogbn_dataset
from memory_usage import see_memory_usage, nvidia_smi_usage
import tracemalloc
from cpu_mem_usage import get_memory
from statistics import mean

from my_utils import parse_results
from collections import Counter

import pickle
from utils import Logger
import os 
import numpy
import pdb



def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.device >= 0:
		torch.cuda.manual_seed_all(args.seed)
		torch.cuda.manual_seed(args.seed)
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.deterministic = True
		dgl.seed(args.seed)
		dgl.random.seed(args.seed)





def get_compute_num_nids(blocks):
	res=0
	for b in blocks:
		res+=len(b.srcdata['_ID'])
	return res


def get_FL_output_num_nids(blocks):

	output_fl =len(blocks[0].dstdata['_ID'])
	return output_fl


def knapsack_float(items, capacity):
    n = len(items)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        value, weight = items[i - 1]
        for j in range(capacity + 1):
            if weight <= j:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][int(j - weight)] + value)
            else:
                dp[i][j] = dp[i - 1][j]
	# Find the optimal items
    optimal_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            optimal_items.append(i - 1)
            w -= items[i - 1][1]
            w = int(w)
    return dp[-1][-1], optimal_items


def EST_mem(modified_mem, optimal_items):
    # print(modified_mem)
    # print(optimal_items)
    result = 0
    for idx, ll in enumerate(modified_mem):
        if idx in optimal_items:
            result += ll[1]

    return result
    
    


def knapsack(items, capacity):
    n = len(items)
    # Initialize the dynamic programming table
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    # Fill the table using dynamic programming
    for i in range(1, n + 1):
        item_value, item_weight = items[i - 1]
        for w in range(capacity + 1):
            if item_weight <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - item_weight] + item_value)
            else:
                dp[i][w] = dp[i - 1][w]

    # Find the optimal items
    optimal_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            optimal_items.append(i - 1)
            w -= items[i - 1][1]

    return dp[n][capacity], optimal_items

# # Example usage:
# items = [(60, 10), (100, 20), (120, 30)]  # (value, weight)
# capacity = 50
# max_value, optimal_items = knapsack(items, capacity)
# print("Maximum value:", max_value)
# print("Optimal items:", optimal_items)
def print_mem(list_mem):
    deg = 1
    for item in list_mem:
        print('degree '+str(deg) +' '+str(item[0]))
        deg += 1
    print()
    
def estimate_mem(data_dict, in_feat, hidden_size, redundant_ratio, fanout):	
	
	estimated_mem_dict = {}
	for batch_id, data in enumerate(data_dict):
		
		batch_est_mem = 0
		for index, layer in enumerate(data):
			for key, value in layer.items():
				if index == 0:  # For first layer
					batch_est_mem += key * value * in_feat * 18 * 4 / 1024 / 1024 / 1024
				else:  # For second and third layer
					batch_est_mem += key * value * hidden_size * 18 *4 / 1024 / 1024 / 1024

		estimated_mem_dict[batch_id] = batch_est_mem
	print('estimated_mem_dict')
	print(estimated_mem_dict)
	print(list(estimated_mem_dict.values())[:-1])
	print()
	modified_estimated_mem_list = []
	for idx,(key, val) in enumerate(estimated_mem_dict.items()):
		# modified_estimated_mem_list.append(estimated_mem_dict[key]*redundant_ratio[idx]) 
		# # redundant_ratio[i] is a variable depends on graph characteristic
		# print(' MM estimated memory/GB degree '+str(key)+': '+str(estimated_mem_dict[key]) + " * " +str(redundant_ratio[idx])  ) 
		# modified_estimated_mem_list.append(estimated_mem_dict[key]*redundant_ratio[idx]*0.226/2) 
		# print(' MM estimated memory/GB degree '+str(key)+': '+str(estimated_mem_dict[key]) + " * " +str(redundant_ratio[idx]) +"*"+str(0.226/2) ) 
		modified_estimated_mem_list.append(estimated_mem_dict[key]*min(redundant_ratio[idx]*0.226,1)) 
		print(' MM estimated memory/GB degree '+str(key)+': '+str(estimated_mem_dict[key]) + " * min( " +str(redundant_ratio[idx]) +"*"+str(0.226)+',1') 
	
	print()
	print('modified_estimated_mem_list ')
	print(modified_estimated_mem_list)
	print()
	
	return modified_estimated_mem_list, list(estimated_mem_dict.values())

# def swap(split_list):
# 	index1 = 0
# 	index2 = len(split_list)-1
# 	temp = split_list[index1]
# 	split_list[index1] = split_list[index2]
# 	split_list[index2] = temp
# 	return split_list



# def split_tensor(tensor, num_parts):
# 	N = tensor.size(0)
# 	split_size = N // num_parts
# 	if N % num_parts != 0:
# 		split_size += 1

# 	# Split the tensor into two parts
# 	split_tensors = torch.split(tensor, split_size)

# 	# Convert the split tensors into a list
# 	split_list = list(split_tensors)
 
# 	split_list = swap(split_list)
	
# 	weight_list = [len(part) / N for part in split_tensors]
# 	return split_list, weight_list

# def dataloader_gen(full_batch_dataloader,g,processed_fan_out, num_batch):
# 	block_dataloader = []
# 	blocks_list=[]
# 	weights_list=[]
	
# 	for step , (src, dst, full_blocks) in enumerate(full_batch_dataloader):
		
# 		dst_list, weights_list = split_tensor(dst, num_batch) #######
# 		final_dst_list = dst_list
# 		pre_dst_list=[]
# 		for layer , full_block in enumerate(reversed(full_blocks)):
# 			layer_block_list=[]
			
# 			layer_graph = dgl.edge_subgraph(g, full_block.edata['_ID'],relabel_nodes=False,store_ids=True)
# 			src_len = len(full_block.srcdata['_ID'])
# 			layer_graph.ndata['_ID']=torch.tensor([-1]*len(layer_graph.nodes()))
# 			layer_graph.ndata['_ID'][:src_len] = full_block.srcdata['_ID']

# 			if layer == 0:
# 				print('the output layer ')
# 				for i,dst_new in enumerate(dst_list) :
# 					sg1 = dgl.sampling.sample_neighbors_range(layer_graph, dst_new, processed_fan_out[-1])
# 					block = dgl.to_block(sg1,dst_new, include_dst_in_src= True)
# 					pre_dst_list.append(block.srcdata[dgl.NID]) 
# 					layer_block_list.append(block)
# 			elif layer == 1:
# 				print('input layer')
# 				src_list=[]
# 				for i,dst_new in enumerate(pre_dst_list):
# 					sg1 = dgl.sampling.sample_neighbors_range(layer_graph, dst_new, processed_fan_out[0])
# 					block = dgl.to_block(sg1,dst_new, include_dst_in_src= True)
# 					layer_block_list.append(block)
# 					src_list.append(block.srcdata[dgl.NID]) 
# 				final_src_list = src_list

# 			blocks_list.append(layer_block_list)

# 		blocks_list = blocks_list[::-1]
# 		for batch_id in range(num_batch):
# 			cur_blocks = [blocks[batch_id] for blocks in blocks_list]
# 			dst = final_dst_list[batch_id]
# 			src = final_src_list[batch_id]
# 			block_dataloader.append((src, dst, cur_blocks))
# 	return block_dataloader, weights_list


#### Entry point
def run(args, device, data):
	if args.GPUmem:
		see_memory_usage("----------------------------------------start of run function ")
	# Unpack data
	g, nfeats, labels, n_classes, train_nid, val_nid, test_nid = data
	in_feats = len(nfeats[0])
	print('in feats: ', in_feats)
	# nvidia_smi_list=[]

	if args.selection_method =='metis':
		args.o_graph = dgl.node_subgraph(g, train_nid)


	# sampler = dgl.dataloading.MultiLayerNeighborSampler(
	# 	[int(fanout) for fanout in args.fan_out.split(',')])
	# full_batch_size = len(train_nid)
	fan_out_list = [fanout for fanout in args.fan_out.split(',')]
	fan_out_list = ' '.join(fan_out_list).split()
	processed_fan_out = [int(fanout) for fanout in fan_out_list] # remove empty string


	args.num_workers = 0


	model = GraphSAGE(
					in_feats,
					args.num_hidden,
					n_classes,
					args.aggre,
					args.num_layers,
					F.relu,
					args.dropout).to(device)

	loss_fcn = nn.CrossEntropyLoss()

	# if args.GPUmem:
	# 	see_memory_usage("----------------------------------------after model to device")
	logger = Logger(args.num_runs, args)
	for run in range(args.num_runs):
		model.reset_parameters()
		# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		for epoch in range(args.num_epochs):
			model.train()

			loss_sum=0
			# start of data preprocessing part---s---------s--------s-------------s--------s------------s--------s----
			if args.load_full_batch:
				full_batch_dataloader=[]
				file_name=r'/home/cc/dataset/fan_out_'+args.fan_out+'/'+args.dataset+'_'+str(epoch)+'_items.pickle'
				with open(file_name, 'rb') as handle:
					item=pickle.load(handle)
					full_batch_dataloader.append(item)
			
			if args.num_batch > 1:
				time0 = time.time()
				block_dataloader, weights_list= dataloader_gen_range(full_batch_dataloader,g,processed_fan_out, args.num_batch)
				time1 = time.time()
				data_dict = []
				print('redundancy ratio #input/#seeds/degree')
				redundant_ratio = []
				for step, (input_nodes, seeds, blocks) in enumerate(block_dataloader):
					print(len(input_nodes)/len(seeds)/(step+1))
					redundant_ratio.append(len(input_nodes)/len(seeds)/(step+1))
    
				time_dict_start = time.time()
				for step, (input_nodes, seeds, blocks) in enumerate(block_dataloader):
					layer = 0
					dict_list =[]
					for b in blocks:
						print('layer ', layer)
						graph_in = dict(Counter(b.in_degrees().tolist()))
						graph_in = dict(sorted(graph_in.items()))

						print(graph_in)
						dict_list.append(graph_in)

						layer = layer +1
					print()
					data_dict.append(dict_list)
				time_dict_end = time.time()
    
				print('data_dict')
				print(data_dict)
				fanout_list = [int(fanout) for fanout in args.fan_out.split(',')]
				fanout = fanout_list[-1]
				time_est_start = time.time()
				modified_res, res = estimate_mem(data_dict, in_feats, args.num_hidden, redundant_ratio, fanout)
				time_est_end = time.time()
				fanout_list = [int(fanout) for fanout in args.fan_out.split(',')]
				fanout = fanout_list[1]
				print('modified_mem [1, fanout-1]: ' )
				print(modified_res[:fanout-1])
				print('the sum of modified_mem [1, fanout-1]: ', sum(modified_res[:fanout-1]))
				print('mem size of fanout degree bucket by formula (GB): ', res[fanout-1])
				print()
				print('the modified memory estimation spend (sec)', time.time()-time1)
				print('the time of number of fanout blocks generation (sec)', time1-time0)

				print('the time dict collection (sec)', time_dict_end - time_dict_start)
				print('the time estimate mem (sec)', time_est_end - time_est_start)
				
					

def main():
	# get_memory("-----------------------------------------main_start***************************")
	tt = time.time()
	print("main start at this time " + str(tt))
	argparser = argparse.ArgumentParser("multi-gpu training")
	argparser.add_argument('--device', type=int, default=0,
		help="GPU device ID. Use -1 for CPU training")
	argparser.add_argument('--seed', type=int, default=1236)
	argparser.add_argument('--setseed', type=bool, default=True)
	argparser.add_argument('--GPUmem', type=bool, default=True)
	argparser.add_argument('--load-full-batch', type=bool, default=True)
	# argparser.add_argument('--root', type=str, default='../my_full_graph/')
	argparser.add_argument('--dataset', type=str, default='ogbn-arxiv')
	# argparser.add_argument('--dataset', type=str, default='ogbn-mag')
	# argparser.add_argument('--dataset', type=str, default='ogbn-products')
	# argparser.add_argument('--dataset', type=str, default='cora')
	# argparser.add_argument('--dataset', type=str, default='karate')
	# argparser.add_argument('--dataset', type=str, default='reddit')
	# argparser.add_argument('--aggre', type=str, default='mean')
	argparser.add_argument('--aggre', type=str, default='lstm')
	argparser.add_argument('--model', type=str, default='graphsage')
	# argparser.add_argument('--selection-method', type=str, default='range_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='random_bucketing')
	argparser.add_argument('--selection-method', type=str, default='fanout_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='custom_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='__bucketing')
	# argparser.add_argument('--num-batch', type=int, default=30)
	argparser.add_argument('--num-batch', type=int, default=25)
	# argparser.add_argument('--num-layers', type=int, default=3)
	# argparser.add_argument('--fan-out', type=str, default='10,25,30')
	argparser.add_argument('--mem-constraint', type=float, default=18.1)
	# argparser.add_argument('--num-hidden', type=int, default=256)
	argparser.add_argument('--num-hidden', type=int, default=256)

	argparser.add_argument('--num-runs', type=int, default=1)
	argparser.add_argument('--num-epochs', type=int, default=1)


	argparser.add_argument('--num-layers', type=int, default=2)
	argparser.add_argument('--fan-out', type=str, default='10,25')

	# argparser.add_argument('--num-layers', type=int, default=3)
	# argparser.add_argument('--fan-out', type=str, default='10,25,30')

	argparser.add_argument('--log-indent', type=float, default=3)
#--------------------------------------------------------------------------------------

	argparser.add_argument('--lr', type=float, default=1e-3)
	argparser.add_argument('--dropout', type=float, default=0.5)
	argparser.add_argument("--weight-decay", type=float, default=5e-4,
						help="Weight for L2 loss")
	argparser.add_argument("--eval", action='store_true', 
						help='If not set, we will only do the training part.')

	argparser.add_argument('--num-workers', type=int, default=4,
		help="Number of sampling processes. Use 0 for no extra process.")
	

	args = argparser.parse_args()
	if args.setseed:
		set_seed(args)
	device = "cpu"
	if args.GPUmem:
		see_memory_usage("-----------------------------------------before load data ")
	if args.dataset=='karate':
		g, n_classes = load_karate()
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='cora':
		g, n_classes = load_cora()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='pubmed':
		g, n_classes = load_pubmed()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='reddit':
		g, n_classes = load_reddit()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
	elif args.dataset == 'ogbn-arxiv':
		data = load_ogbn_dataset(args.dataset,  args)
		device = "cuda:0"

	elif args.dataset=='ogbn-products':
		g, n_classes = load_ogb(args.dataset,args)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='ogbn-mag':
		# data = prepare_data_mag(device, args)
		data = load_ogbn_mag(args)
		device = "cuda:0"
		# run_mag(args, device, data)
		# return
	else:
		raise Exception('unknown dataset')


	best_test = run(args, device, data)


if __name__=='__main__':
	main()