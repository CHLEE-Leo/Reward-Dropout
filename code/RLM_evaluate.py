# %%
import os
from pathlib import Path
import argparse
import json
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import time
import glob
import random
from tqdm import tqdm

import einops
from utils import indice_pad_in_prefix, remove_pad_in_prefix_case, right_pad_after_eos_token, createFolder
from tensorflow.keras.utils import Progbar
from tensorflow.keras import mixed_precision
from transformers import AutoTokenizer, TFAutoModelForCausalLM, TFGPTJForCausalLM, TFT5ForConditionalGeneration, TFBertModel


'''
초기화, 시드지정, 에러표시 함수
'''
def initialize_setting():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


'''
GPU 셋업
'''
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
initialize_setting()
seed_everything(47)

'''
파라미터 로드
'''
parser = argparse.ArgumentParser(description='receive the parameters')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--ref_model_name', type=str, default='opt_large')
parser.add_argument('--rl_model_name', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_epoch', type=int, default=10)
parser.add_argument('--lr', type=float, default=5e-05)
parser.add_argument('--dropout', type=str, required=True)
parser.add_argument('--dropout_rate', type=float, required=True)
parser.add_argument('--decoding', type=str, default='beam')
parser.add_argument('--prefix_ratio', type=float, default=0.15)
parser.add_argument('--gen_len', type=int, default=30)
args = parser.parse_args()

my_dataset = args.dataset
ref_model_name = args.ref_model_name
rl_model_name = args.rl_model_name
my_batch_size = args.batch_size
my_num_epoch = args.num_epoch
my_lr = args.lr
my_dropout = args.dropout
my_dropout_rate = args.dropout_rate
ref_decoding = args.decoding                 # ref_model's decodnig strategy
my_prefix_ratio = args.prefix_ratio         # ratio of prefix to use
my_gen_len = args.gen_len       # length of the sequence 
my_prefix_len = int(my_gen_len * my_prefix_ratio)         # length of prefix (length of prompt)
