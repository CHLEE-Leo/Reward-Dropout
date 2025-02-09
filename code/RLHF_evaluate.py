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
import glob
import csv

import einops
from utils import indice_pad_in_prefix, remove_pad_in_prefix_case, right_pad_after_eos_token, createFolder
from tensorflow.keras.utils import Progbar
from tensorflow.keras import mixed_precision
from transformers import AutoTokenizer, TFAutoModelForCausalLM, TFGPT2LMHeadModel


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

# <eos> 이후의 모든 내용을 삭제하는 함수
def remove_after_eos(text):
    return text.split('<eos>')[0]


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

# my_dataset = 'topic-0'
# ref_model_name = 'opt_large'
# rl_model_name = 'gpt2_small'
# my_batch_size = 256
# my_num_epoch = 5
# my_lr = 5e-06
# my_dropout = 'None'
# my_dropout_rate = 0.0
# ref_decoding = 'stochastic'                 # ref_model's decodnig strategy
# my_prefix_ratio = 0.2         # ratio of prefix to use
# my_gen_len = 30       # length of the sequence 
# my_prefix_len = int(my_gen_len * my_prefix_ratio)         # length of prefix (length of prompt)

'''
각종 경로 설정
'''
# 현 위치의 부모위치 확인
parent_dir = str(Path(os.getcwd()).parents[0])

prep_data_path = parent_dir + '/prep_data' + '/' + my_dataset.split('-')[0] + '/' + rl_model_name       # 데이터 로드 경로 설정 (sentiment-0 과 sentiment-1로 나눌필요 없음)
LOAD_MODEL_DIR = parent_dir + '/weights' + '/' + my_dataset + '/' + rl_model_name + f'/ref={ref_model_name}_ref_dec={ref_decoding}_n_epoch={my_num_epoch}_lr={my_lr}_dropout={my_dropout}_dropout_rate={my_dropout_rate}'                      # 생성결과 저장 경로 설정
SAVE_RESULT_DIR = parent_dir + '/results' + '/' + my_dataset + '/' + rl_model_name + f'/ref={ref_model_name}_ref_dec={ref_decoding}_n_epoch={my_num_epoch}_lr={my_lr}_dropout={my_dropout}_dropout_rate={my_dropout_rate}'                     # 생성결과 저장 경로 설정

'''
테스트 데이터 로드
'''
test_input_dir_list = glob.glob(prep_data_path + '/test_input*')
test_input_x = np.load(test_input_dir_list[0])
prompt_len = int(test_input_x.shape[1] * my_prefix_ratio)

# test_input_att = np.load(test_input_dir_list[1])

test_dataset = tf.data.Dataset.from_tensor_slices(test_input_x[:, :prompt_len])
test_loader = test_dataset.batch(batch_size=my_batch_size)

'''
모델 로드
'''
tokenizer = AutoTokenizer.from_pretrained(
    'gpt2',
    bos_token="<bos>",
    eos_token="<eos>",
    mask_token="[MASK]",
    pad_token="<pad>",
    sep_token="<\s>",
    unk_token="<unk>"

)
model = TFAutoModelForCausalLM.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))
weights_file = tf.train.latest_checkpoint(LOAD_MODEL_DIR) 
model.load_weights(weights_file)
model.trainable = False


'''
생성
'''
for idx, batch in enumerate(tqdm(test_loader)):

    # if idx > 0:
    #     break;

    gen_seqs = model.generate(
        batch,
        max_length=my_gen_len,
    )

    gen_texts = tokenizer.batch_decode(gen_seqs, skip_special_tokens=True)
    gen_texts = [i.replace('\n', '') for i in gen_texts]

    # 리스트의 각 문자열에 대해 함수 적용
    gen_texts = [remove_after_eos(text) for text in gen_texts]

    # dataframe으로 변환
    gen_texts_pd = pd.DataFrame(gen_texts)

    if idx == 0:
        if not os.path.isfile(SAVE_RESULT_DIR):
            gen_texts_pd.to_csv(SAVE_RESULT_DIR+'/gen_texts.csv', header=True, index=False)
        else:
            gen_texts_pd.to_csv(SAVE_RESULT_DIR+'/gen_texts.csv', header=True, index=False)     # 덮어 씌우기
    else:
        gen_texts_pd.to_csv(SAVE_RESULT_DIR+'/gen_texts.csv', mode='a', header=False, index=False)      # 추가하기

'''
정답도 저장
'''
test_texts = tokenizer.batch_decode(test_input_x, skip_special_tokens=True)
test_texts = [i.replace('\n', '') for i in test_texts]
pd.DataFrame(test_texts).to_csv(prep_data_path + '/test_text.csv')