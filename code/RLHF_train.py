# %%
'''
라이브러리 로드
'''
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
import matplotlib.pyplot as plt
import gc

import einops
from utils import createFolder
from tensorflow.keras.utils import Progbar
# from tensorflow.keras import mixed_precision
from transformers import AutoModel, AutoTokenizer, AutoConfig, TFAutoModelForCausalLM, TFGPTJForCausalLM, TFT5ForConditionalGeneration, TFBertModel

# %%
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

    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_global_policy(policy)

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# %%
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
parser.add_argument('--seq_total_len', type=int, default=30)
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
my_seq_total_len = args.seq_total_len       # length of the sequence 
my_prefix_len = int(my_seq_total_len * my_prefix_ratio)         # length of prefix (length of prompt)

# my_dataset = 'topic-1'
# ref_model_name = 'opt_large'
# rl_model_name = 'gpt_j'
# my_batch_size = 64
# my_lr = 5e-06
# my_dropout = 'quantile'
# my_dropout_rate = 0.8
# ref_decoding = 'stochastic'
# my_prefix_ratio = 0.15
# my_seq_total_len = 30
# my_prefix_len = int(my_seq_total_len * my_prefix_ratio)         # length of prefix (length of prompt)


'''
데이터 로드 경로 설정
'''
# 현 위치의 부모위치 확인
parent_dir = str(Path(os.getcwd()).parents[0])

# # 데이터 로드 경로 설정
# data_load_path = parent_dir + '/prep_data' + '/' + my_dataset + '/' + rl_model_name

# %%
'''
Likelihood 스코어 계산 함수
'''
def get_likelihood(gen_texts, all_pred_logits: list, prefix_len: int, gen_seq_len: int):
    '''
    - gen_texts : a batch of generated (sampled) texts in matrix form
        - size : (batch_size, total_seq_len)
    
    - all_pred_logits : 1) a list of all predicted logits in matrix form or 2) a tensor of predicted logits
        - 1) size : [gen_seq_len, (batch_size, vocab_size)]
        - 2) size : (batch_size, gen_seq_len, vocab_size)

    - gen_seq_len = total_seq_len - prefix_len
    '''

    # - [0] 행 갯수, (생성된) 열 갯수 정의
    row_num = gen_texts.shape[0]
    gen_col_num = gen_texts[:, prefix_len:].shape[1]

    # - [1] 확률 텐서 구축
    # - 1-1) all_pred_logits 이 3축 텐서일 경우 (= tp_all_logits)
    if len(all_pred_logits) == gen_texts.shape[0]:
        logit_tensor = copy.deepcopy(all_pred_logits)

    # - 1-2) all_pred_logits 이 3축 텐서가 아닐 경우 (= 리스트일 경우 = bp_all_logits)
    else:   # 즉, len(all_pred_logits) == 15 (= my_gen_len) 일 경우
        list_of_logit_mat = tf.concat(all_pred_logits, axis=0)     # logits matrix list of [seq_len, (batch_size, vocab_size)] dimension
        logit_tensor = einops.rearrange(list_of_logit_mat, '(g_s b) v -> b g_s v', g_s=gen_seq_len)     # logits tensor of (batch_size, gen_seq_len, vocab_size) dimension


    # - [2] 로짓 텐서의 확률 변환 : 로짓 -> 확률 (Likelihood)
    prob_tensor = tf.nn.softmax(logit_tensor, axis = -1)     # probabilities tensor of (batch_size, gen_seq_len, vocab_size) dimension

    # - [3] 모든 행 인덱스 x 모든 (생성된) 열 인덱스의 조합을 long-format 행렬로 정의 
    row_idx_vec = np.arange(row_num)              # batch_size index range
    gen_col_idx_vec = np.arange(gen_col_num)      # gen_seq_len index range
    meshed_row_idx_vec, meshed_gen_col_idx_vec = np.array(np.meshgrid(row_idx_vec, gen_col_idx_vec, indexing='ij'))        # index matrix of [(batch_size x gen_seq_len), 2] dimension

    # - [4] 생성된 시퀀스를 flatten() 하여 모든 token_id 를 차례로 나열한 벡터로 변환
    # -     참고로, token_id = vocab_idx
    vocab_idx_vec = tf.reshape(gen_texts[:, prefix_len:], shape=(-1, ))      # sampled tokens vector of (batch_size x gen_seq_len) size

    # - [5] 각 행이 "row_idx", "col_idx", "vocab_idx (= token_id)" 의 정보를 담고 있도록,
    # -     [(row_num x gen_col_num), 3] 차원의 행렬 구축
    full_token_idx_matrix = np.array([meshed_row_idx_vec.flatten(), meshed_gen_col_idx_vec.flatten(), vocab_idx_vec]).T     # sampled tokens by index matrix of [(batch_size x gen_seq_len), 3] dimension

    # - [6] (batch_size, gen_seq_len, vocab_size) 차원의 prob_tensor 행렬에서
    # -     full_token_idx_matrix 에 담긴 위치에 존재하는 값들을 gather 하여,
    # -     (batch_size x gen_seq_len) 차원의 gathered_prob_vec 를 구축
    gathered_prob_vec = tf.gather_nd(params = prob_tensor, indices = full_token_idx_matrix)     # gather probabilities (logits) of tokens sampled with repetition_penelty
 
    # - [7] gathered_prob_vec 를 (batch_size, gen_seq_len) 차원의 gathered_prob_mat 로 차원 변환
    gathered_prob_mat = einops.rearrange(gathered_prob_vec, '(b g_s) -> b g_s', g_s=gen_seq_len)     # rearrange the probabilities (logits) matrix by (batch_size, gen_seq_len)

    # - [8] gathered_prob_mat 에서 gen_seq_len 차원을 따라 평균값을 반환하며 차원 축소
    gathered_mean_prob_vec = einops.reduce(gathered_prob_mat, 'b g_s -> b', 'mean')

    return gathered_mean_prob_vec + 1e-07

# @tf.function
# def get_likelihood(gen_texts, all_pred_logits, prefix_len, gen_seq_len):
#     '''
#     - gen_texts : a batch of generated (sampled) texts in matrix form
#         - size : (batch_size, total_seq_len)
    
#     - all_pred_logits : 1) a list of all predicted logits in matrix form or 2) a tensor of predicted logits
#         - 1) size : [gen_seq_len, (batch_size, vocab_size)]
#         - 2) size : (batch_size, gen_seq_len, vocab_size)

#     - gen_seq_len = total_seq_len - prefix_len
#     '''

#     # - [0] 행 갯수, (생성된) 열 갯수 정의
#     row_num = tf.shape(gen_texts)[0]
#     gen_col_num = tf.shape(gen_texts[:, prefix_len:])[1]

#     # - [1] 확률 텐서 구축
#     # - 1-1) all_pred_logits 이 3축 텐서일 경우 (= tp_all_logits)
#     if len(all_pred_logits) == tf.shape(gen_texts)[0]:
#         logit_tensor = tf.identity(all_pred_logits)
#     # - 1-2) all_pred_logits 이 3축 텐서가 아닐 경우 (= 리스트일 경우 = bp_all_logits)
#     else:
#         list_of_logit_mat = tf.concat(all_pred_logits, axis=0)     # logits matrix list of [seq_len, (batch_size, vocab_size)] dimension
#         logit_tensor = einops.rearrange(list_of_logit_mat, '(g_s b) v -> b g_s v', g_s=gen_seq_len)     # logits tensor of (batch_size, gen_seq_len, vocab_size) dimension

#     # - [2] 로짓 텐서의 확률 변환 : 로짓 -> 확률 (Likelihood)
#     prob_tensor = tf.nn.softmax(logit_tensor, axis=-1)     # probabilities tensor of (batch_size, gen_seq_len, vocab_size) dimension

#     # - [3] 모든 행 인덱스 x 모든 (생성된) 열 인덱스의 조합을 long-format 행렬로 정의 
#     row_idx_vec = tf.range(row_num)              # batch_size index range
#     gen_col_idx_vec = tf.range(gen_col_num)      # gen_seq_len index range
#     meshed_row_idx_vec, meshed_gen_col_idx_vec = tf.meshgrid(row_idx_vec, gen_col_idx_vec, indexing='ij')        # index matrix of [(batch_size x gen_seq_len), 2] dimension

#     # - [4] 생성된 시퀀스를 flatten() 하여 모든 token_id 를 차례로 나열한 벡터로 변환
#     vocab_idx_vec = tf.reshape(gen_texts[:, prefix_len:], shape=(-1, ))      # sampled tokens vector of (batch_size x gen_seq_len) size

#     # - [5] 각 행이 "row_idx", "col_idx", "vocab_idx (= token_id)" 의 정보를 담고 있도록,
#     # -     [(row_num x gen_col_num), 3] 차원의 행렬 구축
#     full_token_idx_matrix = tf.stack([tf.reshape(meshed_row_idx_vec, [-1]), 
#                                       tf.reshape(meshed_gen_col_idx_vec, [-1]), 
#                                       vocab_idx_vec], axis=1)     # sampled tokens by index matrix of [(batch_size x gen_seq_len), 3] dimension

#     # - [6] (batch_size, gen_seq_len, vocab_size) 차원의 prob_tensor 행렬에서
#     # -     full_token_idx_matrix 에 담긴 위치에 존재하는 값들을 gather 하여,
#     # -     (batch_size x gen_seq_len) 차원의 gathered_prob_vec 를 구축
#     gathered_prob_vec = tf.gather_nd(params=prob_tensor, indices=full_token_idx_matrix)     # gather probabilities (logits) of tokens sampled with repetition_penelty
 
#     # - [7] gathered_prob_vec 를 (batch_size, gen_seq_len) 차원의 gathered_prob_mat 로 차원 변환
#     gathered_prob_mat = einops.rearrange(gathered_prob_vec, '(b g_s) -> b g_s', g_s=gen_seq_len)     # rearrange the probabilities (logits) matrix by (batch_size, gen_seq_len)

#     # - [8] gathered_prob_mat 에서 gen_seq_len 차원을 따라 평균값을 반환하며 차원 축소
#     gathered_mean_prob_vec = einops.reduce(gathered_prob_mat, 'b g_s -> b', 'mean')

#     return gathered_mean_prob_vec + 1e-07

'''
손실 함수 : CategoricalCrossEntropy (CCE)
'''
sparse_categorical_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')

# @tf.function
def loss_function(real, pred, mask):
    
    # 손실 계산
    losses = sparse_categorical_cross_entropy(real, pred)

    # 마스킹 적용
    mask = tf.cast(mask, dtype = losses.dtype)
    losses *= mask

    # 평균손실 계산
    # return tf.reduce_mean(losses)
    # return losses
    return tf.reduce_mean(losses, axis = -1)[:, tf.newaxis]

'''
정확도 함수
'''
# @tf.function
def accuracy_function(real, pred, mask):
    real = tf.cast(real, dtype = tf.int32)

    # masking (= [MASK] + <pad>)되지 않은 토큰 갯수 세기 
    num_non_masks = tf.reduce_sum(tf.cast(mask, dtype = tf.int32), axis = -1)

    # masking (= [MASK] + <pad>)되지 않은 토큰 갯수 세기 
    non_mask_begin_idx = tf.argmax(mask, axis = -1)[0]

    # 최우도 토큰 반환
    max_pred = tf.argmax(pred, axis = -1)
    max_pred = tf.cast(tf.squeeze(max_pred), dtype = tf.int32)

    # 맞춘 토큰 행렬 (hit_matrix) 구축
    hit_index_mat = tf.cast(tf.where(real == max_pred), dtype = tf.int32)

    if len(hit_index_mat) == 0:
        num_hits = 0
    else:
        # hit_matrix = tf.scatter_nd(hit_index_mat, np.repeat(1, hit_index_mat.shape[0]), shape = real.shape)
        hit_matrix = tf.scatter_nd(indices = hit_index_mat, updates = tf.repeat(1, tf.shape(hit_index_mat)[0]), shape = tf.shape(real))
        num_hits = tf.reduce_sum(hit_matrix[:, non_mask_begin_idx:], axis = -1)            

    # 각 sequence 별로 masking 되지않은 토큰들 중에서 맞춘 비율 계산
    acc = num_hits / num_non_masks
    mean_acc = tf.reduce_mean(acc)
    return mean_acc

'''
보상 함수
'''
# @tf.function
def reward_function(pred_logits: float, true_label: int):
    '''
    예측 logits 값을 받아서 보상으로 반환해주는 함수
    '''
    # logits을 probs로 변환
    pred_probs = tf.nn.softmax(pred_logits, axis = -1)

    # # logits (= probs)이 큰 값에 정답 (1), 작은 값에 오답 (0) 라벨을 부여
    # pred_labels = tf.argmax(pred_logits, axis = -1)

    # # 정오답 라벨을 one_hot 형태로 변환
    # num_labels = len(np.unique(pred_labels))
    # pred_onehot_labels = tf.one_hot(pred_labels, depth=num_labels)

    # # 정오답 라벨을 뒤집고 (1 - one_hot_label), 뒤집힌 값에 해당하는 (즉 오답에 해당하는) 확률을 보상으로 정의
    # # -- 오답이 제어하고자 하는 방향이므로, RL 입장에서는 정답임
    # rewards = tf.reduce_sum((1 - pred_onehot_labels) * pred_probs, axis = -1)[:, tf.newaxis]

    # true_label에 해당하는 확률값을 보상으로 정의
    rewards = pred_probs[:, true_label][:, tf.newaxis]

    return rewards      # rewards : (batch_size, 1)

'''
Dropout_Indicator 클래스
'''
class Dropout_Indicator:
    def __init__(self, **kwargs):
        self.dropout = kwargs['dropout']
        self.dropout_rate = kwargs['dropout_rate']
        self.prev_mean_rl_likelihood = 0.0
    
    def __call__(self, inputs: float):

        inputs = inputs.numpy()       # inputs : (batch_size, 1)

        # random 드롭아웃
        if self.dropout == 'random':

            # dropout_rate 만큼의 샘플을 랜덤하게 뽑아, 해당 샘플들의 reward 드롭아웃
            batch_size = inputs.shape[0]
            dropout_size = int(batch_size * self.dropout_rate)
            dropout_idx = np.random.randint(batch_size, size=dropout_size)
            inputs[dropout_idx] = 0
            dropuot_index = tf.math.not_equal(inputs, 0).numpy()

        # quantile 드롭아웃
        elif self.dropout == 'quantile':
            '''
            - inputs_quantile = np.quantile(inputs, q=dropout_quantile)

            - dropout_rate를 quantile로 간주하고, 해당 quantile 이하에 해당하는 샘플들의 reward 드롭아웃  (Quark의 아이디어 차용)
            '''
            inputs_quantile = np.quantile(inputs, q=self.dropout_rate)
            inputs[inputs < inputs_quantile] = 0
            dropuot_index = tf.math.not_equal(inputs, 0).numpy()
    
        elif self.dropout == 'adaptive':
            inputs[inputs < self.prev_mean_inputs] = 0
            dropuot_index = tf.math.not_equal(inputs, 0).numpy()


        # 드롭아웃 없음
        elif self.dropout == 'None':
            dropuot_index = tf.ones(shape = inputs.shape[0])

        # 그 외 = 에러
        else:
            print('Error! "--dropout" arg must be given by either "None", "random", "quantile", or "adaptive".')

        return dropuot_index


'''
최적화 알고리즘 : Adam Optimizer
'''
optimizers = tf.keras.optimizers.Adam(learning_rate=my_lr)
dropout_params = dict({'dropout' : my_dropout, 'dropout_rate' : my_dropout_rate})
dropout_indicator = Dropout_Indicator(**dropout_params)

# @tf.function
def rl_model_update(data, rl_model):
    '''
    - [참고] ref_likelihood_vec & rl_likelihood_vec = 확률 벡터
    '''
    input_seq, input_mask, target_seq, target_mask, ref_bi_obj_reward_vec, ref_likelihood_vec = data

    with tf.GradientTape() as tape:

        # 예측
        outputs = rl_model(input_seq, attention_mask = input_mask, training = True)
        # print('\noutputs.logits (mean) : {}'.format(tf.reduce_mean(outputs.logits)))
        # print('outputs.logits (max) : {}'.format(tf.reduce_max(outputs.logits)))
        # print('outputs.logits (min) : {}'.format(tf.reduce_min(outputs.logits)))

        # rl_likelihood 기준 샘플 드롭아웃
        rl_likelihood_vec = get_likelihood(target_seq, all_pred_logits=outputs.logits, prefix_len=0, gen_seq_len=target_seq.shape[1])
        # dropout_idx = dropout_indicator(rl_likelihood_vec)
        dropout_idx = dropout_indicator(ref_bi_obj_reward_vec)

        # print('\ndropout_idx (mean) : {}'.format(tf.reduce_mean(dropout_idx)))

        # 중요도 계산
        ref_likelihood_vec += 1e-05                                                             # ref_likelihood_vec는 확률 값인데, 확률값이 0에 가깝게 너무 작으면 또는 0이면 log 씌우거나 나누기 할 때 음의 발산을 해버림. 따라서 이를 방지하고자 1e-05를 더해줌.
        rl_likelihood_vec += 1e-05                                                              # rl_likelihood_vec는 확률 값인데, 확률값이 0에 가깝게 너무 작으면 또는 0이면 log 씌우거나 나누기 할 때 음의 발산을 해버림. 따라서 이를 방지하고자 1e-05를 더해줌.
        importance_vec = (rl_likelihood_vec / ref_likelihood_vec)
        importance_vec = tf.clip_by_norm(importance_vec, clip_norm=1.0)                         # 계산 안정성을 위해 norm_clipping
        # print('\nrl_likelihood_vec (mean) : {}'.format(tf.reduce_mean(rl_likelihood_vec)))
        # print('rl_likelihood_vec (max) : {}'.format(tf.reduce_max(rl_likelihood_vec)))
        # print('rl_likelihood_vec (min) : {}'.format(tf.reduce_min(rl_likelihood_vec)))
        # print('\nref_likelihood_vec (mean) : {}'.format(tf.reduce_mean(ref_likelihood_vec)))
        # print('ref_likelihood_vec (max) : {}'.format(tf.reduce_max(ref_likelihood_vec)))
        # print('ref_likelihood_vec (min) : {}'.format(tf.reduce_min(ref_likelihood_vec)))
        # print('\nimportance_vec (mean) : {}'.format(tf.reduce_mean(importance_vec)))
        
        # 손실 계산
        losses = tf.squeeze(loss_function(real=target_seq, pred=outputs.logits, mask=target_mask))

        # 최종 손실
        losses = tf.cast(losses, dtype=tf.float16)
        importance_vec = tf.cast(importance_vec, dtype=tf.float16)
        rl_likelihood_vec = tf.cast(rl_likelihood_vec, dtype=tf.float16)
        ref_bi_obj_reward_vec = tf.cast(ref_bi_obj_reward_vec, dtype=tf.float16)
        dropout_idx = tf.cast(dropout_idx, dtype=tf.float16)

        # print('\nref_bi_obj_reward_vec (mean) : {}'.format(tf.reduce_mean(ref_bi_obj_reward_vec)))
        # print('\ntf.math.log(rl_likelihood_vec) (mean) : {}'.format(tf.reduce_mean(tf.math.log(rl_likelihood_vec))))

        total_losses = losses * tf.stop_gradient( importance_vec * (ref_bi_obj_reward_vec - tf.math.log(rl_likelihood_vec)) * dropout_idx )
        # total_losses = losses


    # 훈련 가능한 변수들만 선택
    trainable_variables = [var for var in rl_model.trainable_variables if var.trainable]

    # 최적화
    gradients = tape.gradient(total_losses, trainable_variables)
    optimizers.apply_gradients(zip(gradients, trainable_variables))

    # 정확도 계산
    accuracies = accuracy_function(real=target_seq, pred=outputs.logits, mask=target_mask)

    # 그라디언트 및 산출값 삭제
    del gradients, outputs

    return tf.reduce_mean(losses), accuracies, rl_likelihood_vec, dropout_idx
# def rl_model_update(data, rl_model, rl_old_likelihood_vec=None, method='pg'):
#     '''
#     - [참고] ref_likelihood_vec & rl_likelihood_vec = 확률 벡터
#     '''
#     input_seq, input_mask, target_seq, target_mask, ref_bi_obj_reward_vec, ref_likelihood_vec = data

#     with tf.GradientTape() as tape:

#         # 예측
#         outputs = rl_model(input_seq, attention_mask = input_mask, training = True)

#         # rl_likelihood 추정
#         rl_likelihood_vec = get_likelihood(target_seq, all_pred_logits=outputs.logits, prefix_len=0, gen_seq_len=target_seq.shape[1])

#         # 드롭아웃 인덱스 
#         dropout_idx = dropout_indicator(ref_bi_obj_reward_vec)
#         dropout_idx = tf.cast(dropout_idx, dtype=tf.float16)

#         if (method == 'pg') or (method == 'ppo' and rl_old_likelihood_vec == None):
#             # Policy Gradient Method
#             ref_likelihood_vec += 1e-05  # log 씌우거나 나누기 시 음의 발산 방지
#             rl_likelihood_vec += 1e-05
#             importance_vec = (rl_likelihood_vec / ref_likelihood_vec)
#             importance_vec = tf.clip_by_norm(importance_vec, clip_norm=1.0)  # 계산 안정성을 위해 norm_clipping

#             importance_vec = tf.cast(importance_vec, dtype=tf.float16)
#             rl_likelihood_vec = tf.cast(rl_likelihood_vec, dtype=tf.float16)
#             ref_bi_obj_reward_vec = tf.cast(ref_bi_obj_reward_vec, dtype=tf.float16)

#             # 최종 손실
#             losses = tf.cast(tf.squeeze(loss_function(real=target_seq, pred=outputs.logits, mask=target_mask)), dtype=tf.float16)
#             total_losses = losses * tf.stop_gradient(importance_vec * (ref_bi_obj_reward_vec - tf.math.log(rl_likelihood_vec)) * dropout_idx)

#         elif method == 'ppo':
            
#             rl_old_likelihood_vec += 1e-05  # log 씌우거나 나누기 시 음의 발산 방지
#             rl_likelihood_vec += 1e-05

#             importance_vec = (rl_likelihood_vec / ref_likelihood_vec)
#             importance_vec = tf.clip_by_norm(importance_vec, clip_norm=1.0)  # 계산 안정성을 위해 norm_clipping

#             importance_vec = tf.cast(importance_vec, dtype=tf.float16)

#             # PPO 손실 계산
#             ratio = rl_likelihood_vec / rl_old_likelihood_vec
#             clip_param = 0.2
#             clipped_ratio = tf.clip_by_value(ratio, 1 - clip_param, 1 + clip_param)
#             advantage = ref_bi_obj_reward_vec
#             min_advantage = tf.minimum(ratio * advantage, clipped_ratio * advantage)

#             # 최종 손실
#             losses = tf.cast(tf.squeeze(loss_function(real=target_seq, pred=outputs.logits, mask=target_mask)), dtype=tf.float16)
#             total_losses = losses * tf.stop_gradient(importance_vec * min_advantage * dropout_idx)

#         else:
#             raise ValueError(f"Unknown method: {method}")


#     # 정확도 계산
#     accuracies = accuracy_function(real=target_seq, pred=outputs.logits, mask=target_mask)

#     # 훈련 가능한 변수들만 선택
#     trainable_variables = [var for var in rl_model.trainable_variables if var.trainable]

#     # 최적화
#     gradients = tape.gradient(total_losses, trainable_variables)
#     optimizers.apply_gradients(zip(gradients, trainable_variables))

#     return tf.reduce_mean(losses), accuracies, rl_likelihood_vec, dropout_idx


# rl_model = TFAutoModelForCausalLM.from_pretrained(rl_model_weights_dir, from_pt=True)
# print(rl_model.get_input_embeddings().get_weights()[0].shape)
# print(rl_model.get_output_embeddings().get_weights()[0].shape)
# custom_resize_token_embeddings(rl_model, len(rl_model_tokenizer))
from transformers import TFSharedEmbeddings
def custom_resize_token_embeddings(model, new_num_tokens, resize_decoder=False):

    def get_embedding_weight(embedding, input_ids=None):
        if isinstance(embedding, TFSharedEmbeddings):
            if input_ids is None:
                input_ids = tf.range(embedding.vocab_size)
            return embedding(input_ids)

        elif isinstance(embedding, tf.keras.layers.Embedding):
            return embedding.embeddings

        elif isinstance(embedding, tf.keras.layers.Dense):
            return embedding.kernel

        elif isinstance(embedding, tf.Variable):
            return embedding

        else:
            raise ValueError(f"Unsupported embedding type: {type(embedding)}")

    def resize_embedding_layer(embedding, new_num_tokens, input_ids=None):
        if isinstance(embedding, TFSharedEmbeddings):
            old_embedding = get_embedding_weight(embedding, input_ids)
            old_num_tokens, old_embedding_dim = tf.shape(old_embedding)[0], tf.shape(old_embedding)[1]

            size_diff = new_num_tokens - old_num_tokens

            if size_diff > 0:
                # Create new embeddings with zero initialization for the additional tokens
                new_embeddings = tf.concat([old_embedding, tf.zeros([size_diff, old_embedding_dim])], axis=0)
            else:
                # Trim the embeddings if reducing the number of tokens
                new_embeddings = old_embedding[:new_num_tokens]

            return new_embeddings

        else:
            old_embedding = get_embedding_weight(embedding, input_ids)
            old_num_tokens, old_embedding_dim = tf.shape(old_embedding)[1], tf.shape(old_embedding)[0]

            size_diff = new_num_tokens - old_num_tokens

            if size_diff > 0:
                # Create new embeddings with zero initialization for the additional tokens
                new_embeddings = tf.concat([old_embedding, tf.zeros([old_embedding_dim, size_diff])], axis=1)   # new_embeddings = (1024, 256011)
            else:
                # Trim the embeddings if reducing the number of tokens
                new_embeddings = old_embedding[:new_num_tokens]

            return new_embeddings

    dummy_input_ids = tf.range(model.config.vocab_size)

    # Resize input embeddings
    old_input_embeddings = copy.deepcopy(model.get_input_embeddings())
    new_input_embeddings = resize_embedding_layer(old_input_embeddings, new_num_tokens, dummy_input_ids)

    if isinstance(old_input_embeddings, TFSharedEmbeddings):

        if 'embed' in old_input_embeddings.name:
            new_shared_embedding = TFSharedEmbeddings(
                new_num_tokens,
                old_input_embeddings(dummy_input_ids).shape[-1]
            )
            new_shared_embedding.build((new_num_tokens,))
            new_shared_embedding.set_weights([new_input_embeddings])
            model.set_input_embeddings(new_shared_embedding)

        if 'wte' in old_input_embeddings.name:
            model.transformer.wte.weight = tf.Variable(new_input_embeddings)
            model.transformer.wte.vocab_size = new_num_tokens

    else:
        old_input_embeddings.set_weights([new_input_embeddings])

    if resize_decoder == True:
        # Resize output embeddings if they exist
        if model.get_output_embeddings() is not None:
            old_output_embeddings = model.get_output_embeddings()

            if isinstance(old_output_embeddings, tf.keras.layers.Dense):
                old_output_weights = old_output_embeddings.kernel
                new_output_weights = resize_embedding_layer(old_output_weights, new_num_tokens, dummy_input_ids)    # new_embeddings = (1024, 256011)
                old_output_embeddings.kernel.assign(new_output_weights)

                # new_output_embeddings = resize_embedding_layer(old_output_embeddings, new_num_tokens, dummy_input_ids)    # new_embeddings = (1024, 256011)
                # new_shared_embedding = TFSharedEmbeddings(
                #     old_output_embeddings.kernel.shape[0],
                #     new_num_tokens
                # )
                # new_shared_embedding.build((new_num_tokens,))
                # new_shared_embedding.set_weights([new_output_embeddings])
                # model.set_output_embeddings(new_shared_embedding)


                # if old_output_embeddings.bias is not None:
                #     old_output_bias = old_output_embeddings.bias
                #     size_diff = new_num_tokens - tf.shape(old_output_bias)[0]
                #     new_output_bias = tf.concat([old_output_bias, tf.zeros([size_diff])], axis=0) if size_diff > 0 else old_output_bias[:new_num_tokens]
                #     old_output_embeddings.bias.assign(new_output_bias)
            else:
                new_output_embeddings = resize_embedding_layer(old_output_embeddings, new_num_tokens, dummy_input_ids)

                if isinstance(old_output_embeddings, TFSharedEmbeddings):
                    new_shared_embedding = TFSharedEmbeddings(
                        new_num_tokens,
                        old_output_embeddings(dummy_input_ids).shape[-1]
                    )
                    new_shared_embedding.build((new_num_tokens,))
                    new_shared_embedding.set_weights([new_output_embeddings])
                    model.set_output_embeddings(new_shared_embedding)
                else:
                    old_output_embeddings.set_weights(new_output_embeddings)

    # Resize biases if they exist
    if model.get_bias() is not None:
        old_lm_head_bias = model.get_bias()
        new_lm_head_bias = model._v2_get_resized_lm_head_bias(old_lm_head_bias, new_num_tokens)
        model.set_bias(new_lm_head_bias)

    return model.get_input_embeddings()

def add_new_decoding_layer(model, new_num_tokens):

    def build_new_decoding_layer(new_embeddings, new_num_tokens):

        new_num_tokens, new_embedding_dim = new_embeddings.shape[1], new_embeddings.shape[0]
        new_decoding_layer = tf.keras.layers.Dense(units=new_num_tokens, use_bias=False)
        new_decoding_layer.build((new_embedding_dim, ))
        
        # Set weights and then return the new_decoding_layer
        new_decoding_layer.set_weights([new_embeddings])
        return new_decoding_layer

    last_layer = model.get_output_embeddings()
    old_embedding = last_layer.kernel
    old_num_tokens, old_embedding_dim = tf.shape(old_embedding)[1], tf.shape(old_embedding)[0]
    size_diff = new_num_tokens - old_num_tokens

    if size_diff > 0:
        # Create new embeddings with zero initialization for the additional tokens
        new_embeddings = tf.concat([old_embedding, tf.zeros([old_embedding_dim, size_diff])], axis=1)   # new_embeddings = (1024, 256011)
    else:
        # Trim the embeddings if reducing the number of tokens
        new_embeddings = old_embedding[:, :new_num_tokens]

    new_decoding_layer_object = build_new_decoding_layer(new_embeddings, new_num_tokens)

    # new_list_of_layers = old_list_of_layers[:-1] + [new_decoding_layer_object]
    # return new_list_of_layers

    # Update the model's output layer
    model.set_output_embeddings(new_decoding_layer_object)

# import re
# def set_last_layer_trainable(model):
#     # 모델의 모든 변수를 가져오기
#     variables = model.variables

#     # 변수 이름에서 레이어 번호 추출하는 함수
#     def extract_layer_number(variable_name):
#         match = re.search(r'layers\.(\d+)', variable_name)
#         if match:
#             return int(match.group(1))
#         return -1

#     # 변수 이름에서 최대 레이어 번호 찾기
#     max_layer_number = -1
#     for variable in variables:
#         layer_number = extract_layer_number(variable.name)
#         if layer_number > max_layer_number:
#             max_layer_number = layer_number

#     # 최대 레이어 번호를 가진 변수 이름 패턴
#     last_layer_pattern = f'layers.{max_layer_number}'

#     # 모든 변수들을 동결
#     for variable in variables:
#         variable._trainable = False

#     # 마지막 레이어에 해당하는 변수들만 trainable=True로 설정
#     for variable in variables:
#         if last_layer_pattern in variable.name:
#             variable._trainable = True

#     # # 모든 변수들의 이름과 trainable 상태 출력 (필요시 사용)
#     # for variable in variables:
#     #     print(f"Variable name: {variable.name}, trainable: {variable.trainable}")

# %%
'''
REF-pi 모델 로드
'''
# if ref_model_name == 'gpt2_small':
#     '''
#     REF Policy LM
#     - REF Policy LM 에는 .resize_token_embeddings()를 안하는게 확실히 맞음 !
#     '''
#     # config, weight, and tokenizer 로드
#     ref_model_config_dir = parent_dir + '/pretrained_weights' + '/' + ref_model_name
#     ref_model_weights_dir = ref_model_config_dir + '/model'
#     ref_model_tokenizer_dir = ref_model_config_dir + '/tokenizer_right'
#     ref_model_tokenizer = AutoTokenizer.from_pretrained(ref_model_tokenizer_dir)

#     # ref_model 초기화
#     ref_model = TFAutoModelForCausalLM.from_pretrained(ref_model_weights_dir)
#     ref_model.trainable = False
    

# elif ref_model_name == 'gpt2_large':
#     '''
#     REF Policy LM
#     - REF Policy LM 에는 .resize_token_embeddings()를 안하는게 확실히 맞음 !
#     '''
#     # config, weight, and tokenizer 로드
#     ref_model_config_dir = parent_dir + '/pretrained_weights' + '/' + ref_model_name
#     ref_model_weights_dir = ref_model_config_dir + '/model'
#     ref_model_tokenizer_dir = ref_model_config_dir + '/tokenizer_right'
#     ref_model_tokenizer = AutoTokenizer.from_pretrained(ref_model_tokenizer_dir)

#     # ref_model 초기화
#     ref_model = TFAutoModelForCausalLM.from_pretrained(ref_model_weights_dir, from_pt=True)
#     ref_model.trainable = False
    


# elif ref_model_name == 'opt' or ref_model_name == 'opt_large' or ref_model_name == 'xglm' or ref_model_name == 'xglm_large':
#     '''
#     REF Policy LM
#     - REF Policy LM 에는 .resize_token_embeddings()를 안하는게 확실히 맞음 !
#     '''
#     # config, weight, and tokenizer 로드
#     ref_model_config_dir = parent_dir + '/pretrained_weights' + '/' + ref_model_name
#     ref_model_weights_dir = ref_model_config_dir + '/model'
#     ref_model_tokenizer_dir = ref_model_config_dir + '/tokenizer_right'       # encoder-decoder 모델은 tokenizer_right 아닐껄..?
#     ref_model_tokenizer = AutoTokenizer.from_pretrained(ref_model_tokenizer_dir)

#     # ref_model 초기화
#     ref_model = TFAutoModelForCausalLM.from_pretrained(ref_model_weights_dir, from_pt=True)
#     ref_model.trainable = False


# elif ref_model_name == 'gpt_j':
#     '''
#     REF Policy LM
#     - REF Policy LM 에는 .resize_token_embeddings()를 안하는게 확실히 맞음 !
#     '''
#     # config, weight, and tokenizer 로드
#     ref_model_config_dir = parent_dir + '/pretrained_weights' + '/' + ref_model_name
#     ref_model_weights_dir = ref_model_config_dir + '/model'
#     ref_model_tokenizer_dir = ref_model_config_dir + '/tokenizer_right'
#     ref_model_tokenizer = AutoTokenizer.from_pretrained(ref_model_tokenizer_dir)

#     # ref_model 초기화
#     ref_model = TFGPTJForCausalLM.from_pretrained(ref_model_weights_dir, from_pt=True)
#     ref_model.trainable = False


# elif ref_model_name == 't5' or ref_model_name == 'flan_t5' or ref_model_name == 'flan_ul2':
#     '''
#     REF Policy LM
#     - REF Policy LM 에는 .resize_token_embeddings()를 안하는게 확실히 맞음 !
#     '''

#     # config, weight, and tokenizer 로드
#     ref_model_config_dir = parent_dir + '/pretrained_weights' + '/' + ref_model_name
#     ref_model_weights_dir = ref_model_config_dir + '/model'
#     ref_model_tokenizer_dir = ref_model_config_dir + '/tokenizer_right'
#     ref_model_tokenizer = AutoTokenizer.from_pretrained(ref_model_tokenizer_dir)

#     # ref_model 초기화010
#     ref_model = TFT5ForConditionalGeneration.from_pretrained(ref_model_weights_dir, from_pt=True)
#     ref_model.resize_token_embeddings(len(ref_model_tokenizer))

'''
RL-pi 모델 로드
'''
if rl_model_name == 'gpt2_small_no_init':

    # config, weight, and tokenizer 로드
    rl_model_config_dir = parent_dir + '/pretrained_weights' + '/' + rl_model_name.replace('_no_init', '')
    # rl_model_weights_dir = rl_model_config_dir + '/model'
    rl_model_tokenizer_dir = rl_model_config_dir + '/tokenizer_right'
    rl_model_tokenizer = AutoTokenizer.from_pretrained(rl_model_tokenizer_dir)
    
    # rl_model 초기화 without initialization
    rl_model = TFAutoModelForCausalLM.from_config(AutoConfig.from_pretrained("gpt2"))
    rl_model.resize_token_embeddings(len(rl_model_tokenizer))

elif rl_model_name == 'gpt2_small':

    # config, weight, and tokenizer 로드
    rl_model_config_dir = parent_dir + '/pretrained_weights' + '/' + rl_model_name
    rl_model_weights_dir = rl_model_config_dir + '/model'
    rl_model_tokenizer_dir = rl_model_config_dir + '/tokenizer_right'
    rl_model_tokenizer = AutoTokenizer.from_pretrained(rl_model_tokenizer_dir)
    
    # rl_model 초기화
    rl_model = TFAutoModelForCausalLM.from_pretrained(rl_model_weights_dir)
    rl_model.resize_token_embeddings(len(rl_model_tokenizer))

elif rl_model_name == 'gpt2_large':
    
    # config, weight, and tokenizer 로드
    rl_model_config_dir = parent_dir + '/pretrained_weights' + '/' + rl_model_name
    rl_model_weights_dir = rl_model_config_dir + '/model'
    rl_model_tokenizer_dir = rl_model_config_dir + '/tokenizer_right'
    rl_model_tokenizer = AutoTokenizer.from_pretrained(rl_model_tokenizer_dir)
    
    # rl_model 초기화
    rl_model = TFAutoModelForCausalLM.from_pretrained(rl_model_weights_dir, from_pt=True)
    rl_model.resize_token_embeddings(len(rl_model_tokenizer))


elif rl_model_name == 'opt':

    # config, weight, and tokenizer 로드
    rl_model_config_dir = parent_dir + '/pretrained_weights' + '/' + rl_model_name
    rl_model_weights_dir = rl_model_config_dir + '/model'
    rl_model_tokenizer_dir = rl_model_config_dir + '/tokenizer_right'       # encoder-decoder 모델은 tokenizer_right 아닐껄..?
    rl_model_tokenizer = AutoTokenizer.from_pretrained(rl_model_tokenizer_dir)

    # rl_model 초기화
    rl_model = TFAutoModelForCausalLM.from_pretrained(rl_model_weights_dir, from_pt=True)
    rl_model.resize_token_embeddings(len(rl_model_tokenizer))

elif rl_model_name == 'xglm':

    # config, weight, and tokenizer 로드
    rl_model_config_dir = parent_dir + '/pretrained_weights' + '/' + rl_model_name
    rl_model_weights_dir = rl_model_config_dir + '/model'
    rl_model_tokenizer_dir = rl_model_config_dir + '/tokenizer_right'       # encoder-decoder 모델은 tokenizer_right 아닐껄..?
    rl_model_tokenizer = AutoTokenizer.from_pretrained(rl_model_tokenizer_dir)

    # rl_model 초기화
    rl_model = TFAutoModelForCausalLM.from_pretrained(rl_model_weights_dir, from_pt=True)
    # rl_model.resize_token_embeddings(len(rl_model_tokenizer))

    # 토큰 임베딩 1 (encoder만)
    custom_resize_token_embeddings(rl_model, len(rl_model_tokenizer), resize_decoder=False)

    # 토큰 임베딩 2 (decoder만; 임베딩 레이어 아예 새로 정의)
    # rl_model.layers = add_new_decoding_layer(rl_model.layers, len(rl_model_tokenizer))
    add_new_decoding_layer(rl_model, len(rl_model_tokenizer))

elif rl_model_name == 'xglm_large' or rl_model_name == 'gpt_j':

    # config, weight, and tokenizer 로드
    rl_model_config_dir = parent_dir + '/pretrained_weights' + '/' + rl_model_name
    rl_model_weights_dir = rl_model_config_dir + '/model'
    rl_model_tokenizer_dir = rl_model_config_dir + '/tokenizer_right'       # encoder-decoder 모델은 tokenizer_right 아닐껄..?
    rl_model_tokenizer = AutoTokenizer.from_pretrained(rl_model_tokenizer_dir)

    # rl_model 초기화
    rl_model = TFAutoModelForCausalLM.from_pretrained(rl_model_weights_dir, from_pt=True)
    # rl_model.resize_token_embeddings(len(rl_model_tokenizer))

    # 토큰 임베딩 1 (encoder만)
    custom_resize_token_embeddings(rl_model, len(rl_model_tokenizer), resize_decoder=False)

    # 토큰 임베딩 2 (decoder만; 마지막 레이어 (= 임베딩 레이어)를 아예 새로 정의)
    # rl_model.layers = add_new_decoding_layer(rl_model.layers, len(rl_model_tokenizer))
    add_new_decoding_layer(rl_model, len(rl_model_tokenizer))

    # 마지막 레이어 제외 전부 동결
    # set_last_layer_trainable(rl_model)

    # 모든 레이어 동결
    for layer in rl_model.layers:
        layer.trainable = False

    # GPT-J 모델의 마지막 레이어 동결 해제
    if 'gpt_j' in rl_model_name:
        rl_model.transformer.h[-1].trainable = True

    # XGLM 모델의 마지막 레이어 동결 해제
    elif 'xglm_large' in rl_model_name:
        rl_model.model.layers[-1].trainable = True


elif rl_model_name == 'opt_large':

    # config, weight, and tokenizer 로드
    rl_model_config_dir = parent_dir + '/pretrained_weights' + '/' + rl_model_name
    rl_model_weights_dir = rl_model_config_dir + '/model'
    rl_model_tokenizer_dir = rl_model_config_dir + '/tokenizer_right'       # encoder-decoder 모델은 tokenizer_right 아닐껄..?
    rl_model_tokenizer = AutoTokenizer.from_pretrained(rl_model_tokenizer_dir)

    # rl_model 초기화
    rl_model = TFAutoModelForCausalLM.from_pretrained(rl_model_weights_dir, from_pt=True)
    rl_model.resize_token_embeddings(len(rl_model_tokenizer))
    
    # 마지막 레이어 제외 전부 동결
    # set_last_layer_trainable(rl_model)

    # 모든 레이어 동결
    for layer in rl_model.layers:
        layer.trainable = False

    # OPT 모델의 마지막 레이어 동결 해제
    if 'opt_large' in rl_model_name:
        rl_model.model.decoder.layers[-1].trainable = True


# elif rl_model_name == 'gpt_j':

#     # config, weight, and tokenizer 로드
#     rl_model_config_dir = parent_dir + '/pretrained_weights' + '/' + rl_model_name
#     rl_model_weights_dir = rl_model_config_dir + '/model'
#     rl_model_tokenizer_dir = rl_model_config_dir + '/tokenizer_right'
#     rl_model_tokenizer = AutoTokenizer.from_pretrained(rl_model_tokenizer_dir)

#     # rl_model 초기화
#     rl_model = TFGPTJForCausalLM.from_pretrained(rl_model_weights_dir, from_pt=True)
#     rl_model.resize_token_embeddings(len(rl_model_tokenizer))


# elif rl_model_name == 't5' or rl_model_name == 'flan_t5' or rl_model_name == 'flan_ul2':

#     # config, weight, and tokenizer 로드
#     rl_model_config_dir = parent_dir + '/pretrained_weights' + '/' + rl_model_name
#     rl_model_weights_dir = rl_model_config_dir + '/model'
#     rl_model_tokenizer_dir = rl_model_config_dir + '/tokenizer_right'
#     rl_model_tokenizer = AutoTokenizer.from_pretrained(rl_model_tokenizer_dir)

#     # rl_model 초기화
#     rl_model = TFT5ForConditionalGeneration.from_pretrained(rl_model_weights_dir, from_pt=True)
#     rl_model.resize_token_embeddings(len(rl_model_tokenizer))


# %%
'''
버트 모델 마지막 레이어 추가를 위해 클래스 정의
'''
class BERT_Classifier(tf.keras.Model):
    def __init__(self, bert_model, num_labels):
        super(BERT_Classifier, self).__init__()
        self.bert_model = bert_model
        self.num_labels = num_labels
        self.dropout = tf.keras.layers.Dropout(self.bert_model.config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(units = self.num_labels, 
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(self.bert_model.config.initializer_range))

    def call(self, data, attention_mask):

        outputs = self.bert_model(data, attention_mask = attention_mask)
        label_outputs = outputs.pooler_output                               # (batch_size, n_dim)
        label_outputs = self.dropout(label_outputs, training=False)         # [중요!] inference에 사용할 BERT_Classifier는 dropout 레이어를 training=False 지정해주어야 함.
        label_preds = self.classifier(label_outputs)                        # (batch_size, num_attris)

        return label_preds

'''
보상함수 모델 (reward function model)
'''
# 2) BERT의 토크나이저 로드 및 모델 초기화 & 미세조정 가중치 로드
# 데이터 셋에 따른 라벨 갯수 설정
if 'emotion' in my_dataset:
    num_labels = 7    

elif 'act' in my_dataset:
    num_labels = 5

elif 'topic' in my_dataset:
    num_labels = 4

else:
    num_labels = 2

# 토크나이저 로드 및 모델 초기화
bert_model_config_dir = parent_dir + '/pretrained_weights/bert'
bert_model_tokenizer_dir = bert_model_config_dir + '/tokenizer'
bert_model_weights_dir = bert_model_config_dir + '/model'
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_tokenizer_dir)
bert_model = TFBertModel.from_pretrained(bert_model_weights_dir)
bert_model.resize_token_embeddings(len(bert_tokenizer))
bert_model = BERT_Classifier(bert_model, num_labels)

# 파인튜닝 가중치 로드
bert_bs = 256
bert_model_finetuned_weights_dir = parent_dir + '/weights' + '/' + my_dataset.split('-')[0] + '/bert'
bert_model_ft_weights_dir = glob.glob(bert_model_finetuned_weights_dir + '/*{}*'.format(bert_bs))[0]
bert_model.load_weights(tf.train.latest_checkpoint(bert_model_ft_weights_dir))
bert_model.trainable = False

# %%
'''
훈련용 데이터 셋팅
'''
# gen_seq_{}.txt 파일들의 경로 리스트
train_gen_seq_dir = glob.glob(parent_dir + '/prep_data' + '/' + my_dataset + '/' + ref_model_name + '/' + '*gen_seq_{}*[0-{}].txt'.format(ref_decoding, my_num_epoch-1))

# likelihood_{}.txt 파일들의 경로 리스트
train_likelihood_dir = glob.glob(parent_dir + '/prep_data' + '/' + my_dataset + '/' + ref_model_name + '/' + '*likelihood_{}*[0-{}].txt'.format(ref_decoding, my_num_epoch-1))

# reward_{}.txt 파일들의 경로 리스트
train_reward_dir = glob.glob(parent_dir + '/prep_data' + '/' + my_dataset + '/' + ref_model_name + '/' + '*reward_{}*[0-{}].txt'.format(ref_decoding, my_num_epoch-1))

'''
가중치 및 생성결과 저장 경로 지정
'''
SAVE_WEIGHT_DIR = parent_dir + '/weights' + '/' + my_dataset + '/' + rl_model_name + '/ref={}_ref_dec={}_n_epoch={}_lr={}_dropout={}_dropout_rate={}'.format(ref_model_name, ref_decoding, my_num_epoch, my_lr, my_dropout, my_dropout_rate)
createFolder(SAVE_WEIGHT_DIR)

SAVE_RESULT_DIR = parent_dir + '/results' + '/' + my_dataset + '/' + rl_model_name + '/ref={}_ref_dec={}_n_epoch={}_lr={}_dropout={}_dropout_rate={}'.format(ref_model_name, ref_decoding, my_num_epoch, my_lr, my_dropout, my_dropout_rate)
createFolder(SAVE_RESULT_DIR)


'''
각 생성 케이스 마다 보상 계산하기
'''
train_loss_history = []
train_acc_history = []
train_reward_history = []
target_label = int(my_dataset.split('-')[-1])
full_target_rewards_list = []

# 훈련 메트릭
# metrics_names = [str(my_model) + '_loss', str(my_model) + '_acc', str(my_model) + '_reward']
total_start_time = time.time()

for gen_case_num, each_case in enumerate(train_gen_seq_dir):

    '''
    [주석처리 하세요.] gen_case_num=0번까지만 확인하는 코드.
    '''
    # if gen_case_num > 0:
    #     break;

    # 각 line이 하나의 element인 list 형식으로 파일 열기
    with open(train_gen_seq_dir[gen_case_num], 'r') as file:
        text_data = file.readlines()
    with open(train_likelihood_dir[gen_case_num], 'r') as file:
        ref_likelihood_data = file.readlines()
    with open(train_reward_dir[gen_case_num], 'r') as file:
        ref_reward_data = file.readlines()
    
    # GPT 토크나이징
    text_data_tokenized = rl_model_tokenizer(text_data, return_tensors='np', truncation=True, padding=True)     # <bos> 포함
    text_data_encoded = text_data_tokenized['input_ids']
    text_data_masks = text_data_tokenized['attention_mask']        

    # 인풋 시퀀스 - 타겟 시퀀스 분리하기
    input_seq = text_data_encoded[:, :-1]
    target_seq = text_data_encoded[:, 1:]
    input_mask = text_data_masks[:, :-1]
    target_mask = text_data_masks[:, 1:]
    # input_seq = text_data_encoded[:128, :-1]
    # target_seq = text_data_encoded[:128, 1:]
    # input_mask = text_data_masks[:128, :-1]
    # target_mask = text_data_masks[:128, 1:]


    # Likelihood 데이터 자료형 변환 : str -> float
    ref_likelihood_data = [float(x) for x in ref_likelihood_data]

    # ref_reward_data 데이터 : str -> float
    ref_reward_data = [float(x) for x in ref_reward_data]

    # ref_likelihood_data = ref_likelihood_data[:128]
    # ref_reward_data = ref_reward_data[:128]

    '''
    텐서 변환 + 배치 슬라이싱
    '''
    with tf.device("/cpu:0"):

        # 텐서 변환
        gen_dat = tf.data.Dataset.from_tensor_slices((input_seq, target_seq, input_mask, target_mask, ref_likelihood_data, ref_reward_data)) # gen_seq랑 reward의 pair를 맞춰주려면 절대로 셔플하면 안됨. 

        # 배치 슬라이싱
        gen_batch = gen_dat.batch(batch_size=my_batch_size, drop_remainder=True)

    '''
    생성 시간 및 진행상황 측정
    '''
    start_time = time.time()
    total_gen_case_num_num = len(train_gen_seq_dir)
    print("\n ref_model : {}, rl_model : {}, \n ref_decoding : {} \n num_gen_case_num : {}/{}".format(ref_model_name, rl_model_name, ref_decoding, gen_case_num + 1, total_gen_case_num_num))
    # pb_i = Progbar(len(gen_batch), stateful_metrics = metrics_names)
    pb_i = Progbar(len(gen_batch))

    '''
    매 gen_caes (= epoch) 마다 훈련 정확도, 손실 및 보상 초기화    
    '''
    train_cumul_acc = 0
    train_cumul_loss = 0
    train_cumul_reward = 0

    '''
    루프
    '''
    for idx, (input_seq, target_seq, input_mask, target_mask, ref_likelihood_vec, ref_reward_vec) in enumerate(gen_batch):

        '''
        [주석처리 하세요.] idx=0번까지만 확인하는 코드.
        '''
        # if idx > 0:
        #     break;

        '''
        모델 예측 및 업데이트
        - train_acc = rl_model의 예측값이 ref_model의 값과 얼마나 matching 되는지 나타내는 지표.
        '''
        ref_bi_obj_reward_vec = ref_reward_vec + tf.math.log(ref_likelihood_vec + 1e-05)     # R + logB 계산 (ref_likelihood_vec는 확률 값인데, 확률값이 0에 가깝게 너무 작으면 log 씌웠을 때 음의 발산을 해버림. 따라서 이를 방지하고자 1e-05를 더해줌.)
        train_loss, train_acc, rl_likelihood, dropout_idx = rl_model_update((input_seq, input_mask, target_seq, target_mask, ref_bi_obj_reward_vec, ref_likelihood_vec), rl_model)
       
        '''
        실시간 훈련성능 계산
        '''
        # online generation from rl_model with "greedy" decoding.
        rl_gen_seq = rl_model.generate(input_seq[:, :my_prefix_len],
                                    attention_mask=input_mask[:, :my_prefix_len],
                                    max_length=my_seq_total_len,
                                    pad_token_id=rl_model_tokenizer.pad_token_id,
                                    eos_token_id=rl_model_tokenizer.eos_token_id, 
                                    do_sample=False)

        # rl_model이 생성한 시퀀스의 보상 계산
        rl_gen_text = rl_model_tokenizer.batch_decode(rl_gen_seq)
        rl_gen_text_bert_encode = bert_tokenizer(rl_gen_text,
                                                 return_tensors='np',
                                                 truncation=True,
                                                 max_length=my_seq_total_len,
                                                 padding=True)
        rl_gen_seq_bert = rl_gen_text_bert_encode['input_ids']
        rl_gen_mask_bert = rl_gen_text_bert_encode['attention_mask']
        rl_gen_reward = bert_model(rl_gen_seq_bert, attention_mask=rl_gen_mask_bert, training=False)
        rl_gen_reward = reward_function(rl_gen_reward, target_label)
        train_reward = tf.reduce_mean(rl_gen_reward)

        '''
        배치별 학습 현황 모니터링 (메트릭 업데이트)
        '''
        # 만약 adaptive 드롭아웃이 아닐시
        if my_dropout != 'adaptive':

            # 진행상태 바 (Progress Bar) 및 메트릭 값 업데이트
            metric_values = [(str(rl_model_name) + '_loss', train_loss), (str(rl_model_name) + '_acc', train_acc), (str(rl_model_name) + '_reward', train_reward)]
            pb_i.update(idx+1, values = metric_values)

        # 만약 adaptive 드롭아웃일 시, batch마다 adaptive 드롭아웃
        else:

            # 하위 10% 경계값 계산
            rl_likelihood_np = rl_likelihood.numpy()
            lower_bound = np.percentile(rl_likelihood_np, 10)

            # 조건에 맞는 값 필터링
            rl_likelihood_np_filtered = rl_likelihood_np[rl_likelihood_np > lower_bound]
            prev_batch_mean_rl_likelihood = tf.reduce_mean(rl_likelihood_np_filtered).numpy()

            # prev_mean_rl_likelihood 업데이트
            Dropout_Indicator.prev_mean_rl_likelihood = prev_batch_mean_rl_likelihood

            # 진행상태 바 (Progress Bar) 및 메트릭 값 업데이트
            metric_values = [(str(rl_model_name) + '_loss', train_loss), (str(rl_model_name) + '_acc', train_acc), (str(rl_model_name) + '_reward', train_reward), ('prev_batch_rl_likelihood (mean)', prev_batch_mean_rl_likelihood)]
            pb_i.update(idx+1, values = metric_values)


        # 배치별 정확도, 손실 및 보상 누계
        train_cumul_acc += train_acc.numpy()
        train_cumul_loss += train_loss.numpy()
        train_cumul_reward += train_reward.numpy()

    # 전체 평균 정확도, 손실 및 보상 (훈련셋)
    train_mean_acc = train_cumul_acc/(idx + 1)
    train_mean_loss = train_cumul_loss/(idx + 1)
    train_mean_reward = train_cumul_reward/(idx + 1)

    # 훈련 손실 히스토리 저장
    train_loss_history += [train_mean_loss]
    loss_history_pd = pd.DataFrame(train_loss_history, columns = ['train_loss'])
    loss_history_pd.to_csv(SAVE_RESULT_DIR + '/loss_history.csv', index_label = 'epoch')

    # 훈련 정확도 히스토리 저장
    train_acc_history += [train_mean_acc]
    acc_history_pd = pd.DataFrame(train_acc_history, columns = ['train_acc'])
    acc_history_pd.to_csv(SAVE_RESULT_DIR + '/acc_history.csv', index_label = 'epoch')

    # 훈련 보상 히스토리 저장
    train_reward_history += [train_mean_reward]
    reward_history_pd = pd.DataFrame(train_reward_history, columns = ['train_reward'])
    reward_history_pd.to_csv(SAVE_RESULT_DIR + '/reward_history.csv', index_label = 'epoch')

    # 가중치 저장 조건
    '''
    test set에 대해서 이전 epoch에서 집계된 최고 성능치보다 현재 epoch의 성능치가 개선될 경우 저장
    '''
    # 현 정확도가 가장 높았던 이전 정확도보다 개선됐을 경우에만 가중치 저장
    rl_model.save_weights(SAVE_WEIGHT_DIR + '/ref_dec={}_epoch={}_weights.ckpt'.format(ref_decoding, gen_case_num))


    # if my_dropout == 'quantile' or my_dropout == 'adaptive':
    #     '''
    #     드롭아웃 히트맵
    #     '''
    #     fig = plt.figure(figsize=(15,0.5))
    #     fig.set_facecolor('white')
        
    #     dropout_rl_likelihood = rl_likelihood * dropout_idx
    #     plt.pcolor(tf.reshape(dropout_rl_likelihood, shape = (1, len(rl_likelihood) ) ) )
    #     # plt.xticks(range(len(map_df.columns)), map_df.columns) ## x축 눈금 생성
    #     # plt.yticks(range(len(map_df.index)), map_df.index) ## y축 눈금 생성
        
    #     plt.colorbar()
    #     plt.tight_layout()
    #     plt.savefig(SAVE_RESULT_DIR + '/dropout_heatmap_{}_epoch.pdf'.format(gen_case_num), format='pdf')
    #     # plt.show()
    if my_dropout == 'quantile' or my_dropout == 'adaptive':
        '''
        드롭아웃 히트맵
        '''
        fig = plt.figure(figsize=(15,0.5))
        fig.set_facecolor('white')
        
        dropout_ref_bi_obj_reward_vec = tf.cast(ref_bi_obj_reward_vec, dtype=tf.float16) * dropout_idx
        dropout_ref_bi_obj_reward_vec = dropout_ref_bi_obj_reward_vec.numpy()
        plt.pcolor(tf.reshape(dropout_ref_bi_obj_reward_vec, shape = (1, len(dropout_ref_bi_obj_reward_vec) ) ) )
        # plt.xticks(range(len(map_df.columns)), map_df.columns) ## x축 눈금 생성
        # plt.yticks(range(len(map_df.index)), map_df.index) ## y축 눈금 생성
        
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(SAVE_RESULT_DIR + '/dropout_heatmap_{}_epoch.pdf'.format(gen_case_num), format='pdf')
        # plt.show()


    # 시간 계산
    end_time = time.time()
    cur_sec = (end_time - start_time)%60
    cur_min = ((end_time - start_time)//60)%60
    cur_hr = ((end_time - start_time)//60)//60
    print("elapsed time : {:.0f} hr, {:.0f} min, {:.2f} sec".format(cur_hr, cur_min, cur_sec))
    total_sec = (end_time - total_start_time)%60
    total_min = ((end_time - total_start_time)//60)%60
    total_hr = ((end_time - total_start_time)//60)//60
    print("total elapsed time : {:.0f} hr, {:.0f} min, {:.2f} sec".format(total_hr, total_min, total_sec))

    gc.collect()
#     # .txt.포맷으로 저장하기 위해 "list of floats -> list of string" 변환하기
#     converted_total_target_rewards_list = ['{:.3f}'.format(x) for x in total_target_rewards_list]

#     # 생성 케이스 별 예측된 보상들 텍스트 파일로 저장하기
#     reward_txt_file_path = SAVE_RESULT_DIR + '/train_reward_{}_{}.txt'.format(ref_decoding, gen_case_num)
#     with open(reward_txt_file_path, 'a') as file:
#         file.write('\n'.join(converted_total_target_rewards_list))

#     # 전체 생성 케이스 통합하기
#     full_target_rewards_list += total_target_rewards_list

# # .txt.포맷으로 저장하기 위해 "list of floats -> list of string" 변환하기
# converted_full_target_rewards_list = ['{:.3f}'.format(x) for x in full_target_rewards_list]

# # 전체 케이스에 대해서 예측된 보상들 텍스트 파일로 저장하기
# reward_txt_file_path = SAVE_RESULT_DIR + '/train_reward_{}_full.txt'.format(ref_decoding)
# with open(reward_txt_file_path, 'a') as file:
#     file.write('\n'.join(converted_full_target_rewards_list))



# %%
# '''
# 드롭아웃 히트맵
# '''
# import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(6,4))
# fig.set_facecolor('white')
 
# plt.pcolor(dropout_rl_likelihood.reshape((16, 16)))
# # plt.xticks(range(len(map_df.columns)), map_df.columns) ## x축 눈금 생성
# # plt.yticks(range(len(map_df.index)), map_df.index) ## y축 눈금 생성
 
# plt.colorbar()
# plt.show()

