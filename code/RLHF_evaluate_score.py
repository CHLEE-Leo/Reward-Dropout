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
# # my_dropout = 'quantile'
# # my_dropout_rate = 0.8
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

REF_DATA_DIR = parent_dir + '/prep_data' + '/' + my_dataset.split('-')[0] + '/' + rl_model_name + '/test_text.csv'    # 데이터 로드 경로 설정 (sentiment-0 과 sentiment-1로 나눌필요 없음)
LOAD_RESULT_DIR = parent_dir + '/results' + '/' + my_dataset + '/' + rl_model_name + f'/ref={ref_model_name}_ref_dec={ref_decoding}_n_epoch={my_num_epoch}_lr={my_lr}_dropout={my_dropout}_dropout_rate={my_dropout_rate}/gen_texts.csv'                     # 생성결과 저장 경로 설정

SAVE_RESULT_DIR = parent_dir + '/results' + '/' + my_dataset + '/' + rl_model_name + f'/ref={ref_model_name}_ref_dec={ref_decoding}_n_epoch={my_num_epoch}_lr={my_lr}_dropout={my_dropout}_dropout_rate={my_dropout_rate}'                     # 

'''
테스트 (reference) 및 생성 데이터 로드
'''
ref_texts = pd.read_csv(REF_DATA_DIR, index_col=0)
gen_texts = pd.read_csv(LOAD_RESULT_DIR)


import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

'''
계산
'''
# nltk 데이터 다운로드
nltk.download('wordnet')
nltk.download('omw-1.4')

# 텍스트 컬럼 추출
ref_texts_list = ref_texts.iloc[:, 0].tolist()
gen_texts_list = gen_texts.iloc[:, 0].tolist()

# nan 값을 포함하는 인덱스 찾기
nan_indices = [i for i, text in enumerate(gen_texts_list) if pd.isna(text)]

# nan 값을 제거한 리스트 생성
ref_texts_list = [text for i, text in enumerate(ref_texts_list) if i not in nan_indices]
gen_texts_list = [text for i, text in enumerate(gen_texts_list) if i not in nan_indices]

# BLEU 점수 계산
bleu_scores = []
smoothie = SmoothingFunction().method4
for ref, gen in zip(ref_texts_list, gen_texts_list):
    ref_split = [ref.split()]
    gen_split = gen.split()
    score = sentence_bleu(ref_split, gen_split, smoothing_function=smoothie)
    bleu_scores.append(score)

average_bleu = sum(bleu_scores) / len(bleu_scores)
print(f'Average BLEU Score: {average_bleu}')

# ROUGE 점수 계산
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
for ref, gen in zip(ref_texts_list, gen_texts_list):
    scores = scorer.score(ref, gen)
    for key in scores:
        rouge_scores[key].append(scores[key].fmeasure)

average_rouge_scores = {key: sum(value) / len(value) for key, value in rouge_scores.items()}
print(f'Average ROUGE Scores: {average_rouge_scores}')

# METEOR 점수 계산
meteor_scores = []
for ref, gen in zip(ref_texts_list, gen_texts_list):
    ref_split = ref.split()
    gen_split = gen.split()
    score = meteor_score([ref_split], gen_split)
    meteor_scores.append(score)

average_meteor = sum(meteor_scores) / len(meteor_scores)
print(f'Average METEOR Score: {average_meteor}')

all_scores = [average_bleu] + list(average_rouge_scores.values()) + [average_meteor]
header_name = ['BLEU', 'ROUGE_1', 'ROUGE_2', 'ROUGE_L', 'METEOR']
results = pd.DataFrame([all_scores,])
results.columns = header_name
results.to_csv(SAVE_RESULT_DIR + '/BLEU_ROUGE_METEOR.csv', header=True)