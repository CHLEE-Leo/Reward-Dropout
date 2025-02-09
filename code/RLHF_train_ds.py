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

import einops
from utils import indice_pad_in_prefix, remove_pad_in_prefix_case, right_pad_after_eos_token, createFolder
from tensorflow.keras.utils import Progbar
import tensorflow_probability as tfp
from tensorflow.keras.mixed_precision import Policy
from transformers import AutoModel, AutoTokenizer, AutoConfig, TFAutoModelForCausalLM, TFGPTJForCausalLM, TFT5ForConditionalGeneration, TFBertModel
# %%
'''
파라미터 로드
'''
parser = argparse.ArgumentParser(description='receive the parameters')
parser.add_argument('--gpus', type=str, required=False, default=None)     # e.g., 2,3 or 0,1 or 0,1,2,3, etc.
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--ref_model_name', type=str, default='opt_large')
parser.add_argument('--rl_model_name', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=5e-05)
parser.add_argument('--dropout', type=str, required=True)
parser.add_argument('--dropout_rate', type=float, required=True)
parser.add_argument('--decoding', type=str, default='beam')
parser.add_argument('--prefix_ratio', type=float, default=0.15)
parser.add_argument('--seq_total_len', type=int, default=30)
args = parser.parse_args()

my_gpus = args.gpus
my_dataset = args.dataset
ref_model_name = args.ref_model_name
rl_model_name = args.rl_model_name
my_batch_size = args.batch_size
my_lr = args.lr
my_dropout = args.dropout
my_dropout_rate = args.dropout_rate
ref_decoding = args.decoding                 # ref_model's decodnig strategy
my_prefix_ratio = args.prefix_ratio         # ratio of prefix to use
my_seq_total_len = args.seq_total_len       # length of the sequence 
my_prefix_len = int(my_seq_total_len * my_prefix_ratio)         # length of prefix (length of prompt)

# my_gpus = '2,3'
# my_dataset = 'sentiment-0'
# ref_model_name = 'opt_large'
# rl_model_name = 'xglm'
# my_batch_size = 2
# my_lr = 5e-06
# my_dropout = 'quantile'
# my_dropout_rate = 0.1
# ref_decoding = 'stochastic'
# my_prefix_ratio = 0.15
# my_seq_total_len = 30
# my_prefix_len = int(my_seq_total_len * my_prefix_ratio)         # length of prefix (length of prompt)

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
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=80000)])  # 40GB 제한
        except RuntimeError as e:
            print(e)

    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


'''
기본 셋업 : Precision, GPU/seed, 분산학습 여부
'''
# # 디버깅을 위해 로그 디바이스 배치를 설정
# tf.debugging.set_log_device_placement(True)

# GPU 및 seed 초기화
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
if my_gpus != None:
    os.environ["CUDA_VISIBLE_DEVICES"] = my_gpus
initialize_setting()
seed_everything(47)

# 분산 전략 설정
# strategy = tf.distribute.MirroredStrategy()
# 사용할 GPU 목록 지정
num_gpus = len(my_gpus.split(','))
gpu_list = []
for i in range(num_gpus):
   gpu_list += ["GPU:{}".format(i)]     # e.g., gpu_list = ["GPU:0", "GPU:1"]  # 필요한 GPU 번호로 설정
strategy = tf.distribute.MirroredStrategy(devices=gpu_list)

# Mixed Precision Training 사용
policy = Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

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
        logit_tensor = tf.identity(all_pred_logits)  # tf.identity 사용: 텐서 복사를 위해 copy.deepcopy 대신 사용

    # - 1-2) all_pred_logits 이 3축 텐서가 아닐 경우 (= 리스트일 경우 = bp_all_logits)
    else:   # 즉, len(all_pred_logits) == 15 (= my_gen_len) 일 경우
        list_of_logit_mat = tf.concat(all_pred_logits, axis=0)     # logits matrix list of [seq_len, (batch_size, vocab_size)] dimension
        logit_tensor = einops.rearrange(list_of_logit_mat, '(g_s b) v -> b g_s v', g_s=gen_seq_len)     # logits tensor of (batch_size, gen_seq_len, vocab_size) dimension

    # - [2] 로짓 텐서의 확률 변환 : 로짓 -> 확률 (Likelihood)
    prob_tensor = tf.nn.softmax(logit_tensor, axis=-1)     # probabilities tensor of (batch_size, gen_seq_len, vocab_size) dimension

    # - [3] 모든 행 인덱스 x 모든 (생성된) 열 인덱스의 조합을 long-format 행렬로 정의 
    # -     tf.range 사용: np.arange 대신 사용하여 TensorFlow 연산으로 변환      
    # -     tf.meshgrid 사용: np.meshgrid 대신 사용하여 TensorFlow 연산으로 변환        
    row_idx_vec = tf.range(row_num, dtype=tf.int32)             # batch_size index range
    gen_col_idx_vec = tf.range(gen_col_num, dtype=tf.int32)     # gen_seq_len index range
    meshed_row_idx_vec, meshed_gen_col_idx_vec = tf.meshgrid(row_idx_vec, gen_col_idx_vec, indexing='ij')          # index matrix of [(batch_size x gen_seq_len), 2] dimension

    # - [4] 생성된 시퀀스를 flatten() 하여 모든 token_id 를 차례로 나열한 벡터로 변환
    # -     참고로, token_id = vocab_idx
    # -     tf.reshape 사용: flatten 대신 사용하여 TensorFlow 연산으로 변환      
    vocab_idx_vec = tf.reshape(gen_texts[:, prefix_len:], shape=(-1,))      # sampled tokens vector of (batch_size x gen_seq_len) size

    # - [5] 모든 인덱스를 int32로 변환하여 데이터 타입 일치
    meshed_row_idx_vec = tf.cast(meshed_row_idx_vec, tf.int32)
    meshed_gen_col_idx_vec = tf.cast(meshed_gen_col_idx_vec, tf.int32)
    vocab_idx_vec = tf.cast(vocab_idx_vec, tf.int32)

    # - [6] 각 행이 "row_idx", "col_idx", "vocab_idx (= token_id)" 의 정보를 담고 있도록,
    # -     [(row_num x gen_col_num), 3] 차원의 행렬 구축
    # -     tf.stack 사용: np.array 대신 사용하여 TensorFlow 연산으로 변환     
    full_token_idx_matrix = tf.stack([tf.reshape(meshed_row_idx_vec, [-1]), tf.reshape(meshed_gen_col_idx_vec, [-1]), vocab_idx_vec], axis=1)       # sampled tokens by index matrix of [(batch_size x gen_seq_len), 3] dimension

    # - [7] (batch_size, gen_seq_len, vocab_size) 차원의 prob_tensor 행렬에서
    # -     full_token_idx_matrix 에 담긴 위치에 존재하는 값들을 gather 하여,
    # -     (batch_size x gen_seq_len) 차원의 gathered_prob_vec 를 구축
    gathered_prob_vec = tf.gather_nd(params=prob_tensor, indices=full_token_idx_matrix)  # gather probabilities (logits) of tokens sampled with repetition_penelty
 
    # - [8] gathered_prob_vec 를 (batch_size, gen_seq_len) 차원의 gathered_prob_mat 로 차원 변환
    gathered_prob_mat = einops.rearrange(gathered_prob_vec, '(b g_s) -> b g_s', g_s=gen_seq_len)  # rearrange the probabilities (logits) matrix by (batch_size, gen_seq_len)

    # - [9] gathered_prob_mat 에서 gen_seq_len 차원을 따라 평균값을 반환하며 차원 축소
    gathered_mean_prob_vec = einops.reduce(gathered_prob_mat, 'b g_s -> b', 'mean')

    return gathered_mean_prob_vec + 1e-07

'''
손실 함수 : CategoricalCrossEntropy (CCE)
'''
sparse_categorical_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')

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
    
    def __call__(self, rl_likelihood: tf.Tensor):

        # random 드롭아웃
        if self.dropout == 'random':

            # dropout_rate 만큼의 샘플을 랜덤하게 뽑아, 해당 샘플들의 reward 드롭아웃
            batch_size = rl_likelihood.shape[0]
            dropout_size = int(batch_size * self.dropout_rate)
            dropout_idx = tf.random.uniform([dropout_size], minval=0, maxval=batch_size, dtype=tf.int32)            # tf.random.uniform 사용: np.random.randint 대신 사용하여 랜덤 인덱스를 생성
            rl_likelihood = tf.tensor_scatter_nd_update(rl_likelihood, tf.expand_dims(dropout_idx, 1), tf.zeros([dropout_size], dtype=rl_likelihood.dtype))            # tf.tensor_scatter_nd_update 사용: 텐서의 특정 인덱스에 대해 값을 업데이트
            dropout_indicator = tf.math.not_equal(rl_likelihood, 0)

        # quantile 드롭아웃
        elif self.dropout == 'quantile':
            '''
            - dropout_rate를 quantile로 간주하고, 해당 quantile 이하에 해당하는 샘플들의 reward 드롭아웃
            '''
            rl_likelihood_quantile = tfp.stats.percentile(rl_likelihood, q=self.dropout_rate*100)            # tfp.stats.percentile 사용: np.quantile 대신 사용하여 퍼센타일 값을 계산
            rl_likelihood = tf.where(rl_likelihood < rl_likelihood_quantile, tf.zeros_like(rl_likelihood), rl_likelihood)            # tf.where 사용: np.where 대신 사용하여 조건에 따라 값을 변경
            dropout_indicator = tf.math.not_equal(rl_likelihood, 0)
    
        elif self.dropout == 'adaptive':
            rl_likelihood = tf.where(rl_likelihood < self.prev_mean_rl_likelihood, tf.zeros_like(rl_likelihood), rl_likelihood)            # tf.where 사용: np.where 대신 사용하여 조건에 따라 값을 변경
            dropout_indicator = tf.math.not_equal(rl_likelihood, 0)

        # 드롭아웃 없음
        elif self.dropout == 'None':
            dropout_indicator = tf.ones(shape=rl_likelihood.shape[0], dtype=tf.bool)            # tf.ones 사용: np.ones 대신 사용하여 동일한 모양의 ones 텐서를 생성

        # 그 외 = 에러
        else:
            print('Error! "--dropout" arg must be given by either "None", "random", "quantile", or "adaptive".')

        return dropout_indicator
    
# %%
'''
RL-pi 모델 초기화 및 옵티마이저, rl_model_update 함수 등을 분산학습 컨텍스트 위에 정의
'''
# with strategy.scope() 블록 내에서 모델, 옵티마이저, step_fn, rl_model_update 정의
# with strategy.scope() 블록 내에서 초기화된 모델과 옵티마이저는 각 GPU에 복제
with strategy.scope():
    '''
    학습모델 초기화
    '''
    # RL 모델 로드
    if rl_model_name == 'gpt2_small_no_init':
        # config, weight, and tokenizer 로드
        rl_model_config_dir = parent_dir + '/pretrained_weights' + '/' + rl_model_name.replace('_no_init', '')
        rl_model_tokenizer_dir = rl_model_config_dir + '/tokenizer_right'
        rl_model_tokenizer = AutoTokenizer.from_pretrained(rl_model_tokenizer_dir)

        # rl_model 초기화
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

    elif rl_model_name in ['opt', 'xglm', 'xglm_large']:
        # config, weight, and tokenizer 로드
        rl_model_config_dir = parent_dir + '/pretrained_weights' + '/' + rl_model_name
        rl_model_weights_dir = rl_model_config_dir + '/model'
        rl_model_tokenizer_dir = rl_model_config_dir + '/tokenizer_right'
        rl_model_tokenizer = AutoTokenizer.from_pretrained(rl_model_tokenizer_dir)

        # rl_model 초기화
        rl_model = TFAutoModelForCausalLM.from_pretrained(rl_model_weights_dir, from_pt=True)
        rl_model.resize_token_embeddings(len(rl_model_tokenizer))

    elif rl_model_name in ['opt_large']:
        # config, weight, and tokenizer 로드
        rl_model_config_dir = parent_dir + '/pretrained_weights' + '/' + rl_model_name
        rl_model_weights_dir = rl_model_config_dir + '/model'
        rl_model_tokenizer_dir = rl_model_config_dir + '/tokenizer_right'
        rl_model_tokenizer = AutoTokenizer.from_pretrained(rl_model_tokenizer_dir)

        # rl_model 초기화
        rl_model = TFAutoModelForCausalLM.from_pretrained(rl_model_weights_dir, from_pt=True)
        rl_model.resize_token_embeddings(len(rl_model_tokenizer))

        # 모든 레이어를 고정(freeze)
        for layer in rl_model.layers:
            layer.trainable = False

        # 마지막 레이어만 trainable 설정
        last_layer = None
        if hasattr(rl_model, 'transformer') and hasattr(rl_model.transformer, 'h'):
            last_layer = rl_model.transformer.h[-1]
        elif hasattr(rl_model, 'decoder') and hasattr(rl_model.decoder, 'layers'):
            last_layer = rl_model.decoder.layers[-1]
        elif hasattr(rl_model, 'gpt_neox') and hasattr(rl_model.gpt_neox, 'layers'):
            last_layer = rl_model.gpt_neox.layers[-1]
        elif hasattr(rl_model, 'transformer') and hasattr(rl_model.transformer, 'layers'):
            last_layer = rl_model.transformer.layers[-1]
        elif hasattr(rl_model, 'layers'):
            last_layer = rl_model.layers[-1]

        if last_layer is not None:
            last_layer.trainable = True
        else:
            raise ValueError("Unsupported model architecture for layer freezing")

    elif rl_model_name == 'gpt_j':
        # config, weight, and tokenizer 로드
        rl_model_config_dir = parent_dir + '/pretrained_weights' + '/' + rl_model_name
        rl_model_weights_dir = rl_model_config_dir + '/model'
        rl_model_tokenizer_dir = rl_model_config_dir + '/tokenizer_right'
        rl_model_tokenizer = AutoTokenizer.from_pretrained(rl_model_tokenizer_dir)

        # rl_model 초기화
        rl_model = TFGPTJForCausalLM.from_pretrained(rl_model_weights_dir, from_pt=True)
        rl_model.resize_token_embeddings(len(rl_model_tokenizer))

    '''
    옵티마이저 (Adam Optimizer) 초기화
    '''
    optimizer = tf.keras.optimizers.Adam(learning_rate=my_lr)
    dropout_params = dict({'dropout': my_dropout, 'dropout_rate': my_dropout_rate})
    dropout_indicator = Dropout_Indicator(**dropout_params)

    '''
    모델 업데이트 함수
    '''
    def rl_model_update(data, rl_model):
        input_seq, input_mask, target_seq, target_mask, ref_bi_obj_reward_vec, ref_likelihood_vec = data

        with tf.GradientTape() as tape:
            # 예측
            outputs = rl_model(input_seq, attention_mask=input_mask, training=True)

            # rl_likelihood 기준 샘플 드롭아웃
            rl_likelihood_vec = get_likelihood(target_seq, all_pred_logits=outputs.logits, prefix_len=0, gen_seq_len=target_seq.shape[1])
            dropout_idx = dropout_indicator(rl_likelihood_vec)

            # 중요도 계산
            ref_likelihood_vec += 1e-05
            ref_likelihood_vec = tf.cast(ref_likelihood_vec, tf.float16)
            rl_likelihood_vec += 1e-05
            importance_vec = (rl_likelihood_vec / ref_likelihood_vec)
            importance_vec = tf.clip_by_norm(importance_vec, clip_norm=1.0)

            # 손실 계산
            losses = tf.squeeze(loss_function(real=target_seq, pred=outputs.logits, mask=target_mask))
            losses = tf.cast(losses, dtype=tf.float16)

            # 정확도 계산
            accuracies = accuracy_function(real=target_seq, pred=outputs.logits, mask=target_mask)

            # 보상 및 중요도 계산
            importance_vec = tf.cast(importance_vec, dtype=tf.float16)
            rl_likelihood_vec = tf.cast(rl_likelihood_vec, dtype=tf.float16)
            ref_bi_obj_reward_vec = tf.cast(ref_bi_obj_reward_vec, dtype=tf.float16)
            dropout_idx = tf.cast(dropout_idx, dtype=tf.float16)

            # 최종 손실
            total_losses = losses * tf.stop_gradient(importance_vec * (ref_bi_obj_reward_vec - tf.math.log(rl_likelihood_vec)) * dropout_idx)

        # opt_large 모델일 때 마지막 레이어만 업데이트, 다른 모델은 전체 업데이트
        if rl_model_name == 'opt_large':
            trainable_vars = [var for layer in rl_model.layers for var in layer.trainable_variables if layer.trainable]
        else:
            trainable_vars = rl_model.trainable_variables

        gradients = tape.gradient(total_losses, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))

        return tf.reduce_mean(losses), accuracies, rl_likelihood_vec, dropout_idx
    
    def train_step(dist_inputs, rl_model):
        per_replica_losses, per_replica_accs, per_replica_rl_likelihood, per_replica_dropout_idx = strategy.run(step_fn, args=(dist_inputs, rl_model))
        
        # PerReplica 객체 처리
        train_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        train_acc = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_accs, axis=None)
        rl_likelihood = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_rl_likelihood, axis=None)
        dropout_idx = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_dropout_idx, axis=None)
        
        return train_loss, train_acc, rl_likelihood, dropout_idx

    '''
    배치 학습 함수
    '''
    @tf.function
    def step_fn(dist_inputs, rl_model):
        input_seq, input_mask, target_seq, target_mask, ref_likelihood_vec, ref_reward_vec = dist_inputs

        ref_bi_obj_reward_vec = ref_reward_vec + tf.math.log(ref_likelihood_vec + 1e-05)
        return rl_model_update((input_seq, input_mask, target_seq, target_mask, ref_bi_obj_reward_vec, ref_likelihood_vec), rl_model)

# 데이터셋 분산
def create_distributed_dataset(strategy, input_seq, target_seq, input_mask, target_mask, ref_likelihood_data, ref_reward_data, batch_size):
    gen_dat = tf.data.Dataset.from_tensor_slices((input_seq, target_seq, input_mask, target_mask, ref_likelihood_data, ref_reward_data))
    gen_batch = gen_dat.batch(batch_size=batch_size, drop_remainder=True)
    dist_gen_batch = strategy.experimental_distribute_dataset(gen_batch)
    return dist_gen_batch

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
train_gen_seq_dir = glob.glob(parent_dir + '/prep_data' + '/' + my_dataset + '/' + ref_model_name + '/' + '*gen_seq_{}*[0-9].txt'.format(ref_decoding))

# likelihood_{}.txt 파일들의 경로 리스트
train_likelihood_dir = glob.glob(parent_dir + '/prep_data' + '/' + my_dataset + '/' + ref_model_name + '/' + '*likelihood_{}*[0-9].txt'.format(ref_decoding))

# reward_{}.txt 파일들의 경로 리스트
train_reward_dir = glob.glob(parent_dir + '/prep_data' + '/' + my_dataset + '/' + ref_model_name + '/' + '*reward_{}*[0-9].txt'.format(ref_decoding))

'''
가중치 및 생성결과 저장 경로 지정
'''
SAVE_WEIGHT_DIR = parent_dir + '/weights' + '/' + my_dataset + '/' + rl_model_name + '/ref={}_ref_dec={}_dropout={}_dropout_rate={}'.format(ref_model_name, ref_decoding, my_dropout, my_dropout_rate)
createFolder(SAVE_WEIGHT_DIR)

SAVE_RESULT_DIR = parent_dir + '/results' + '/' + my_dataset + '/' + rl_model_name + '/ref={}_ref_dec={}_dropout={}_dropout_rate={}'.format(ref_model_name, ref_decoding, my_dropout, my_dropout_rate)
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

# 학습 루프에서 분산된 데이터셋 사용 및 PerReplica 객체 처리
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

    # Likelihood 데이터 자료형 변환 : str -> float
    ref_likelihood_data = [float(x) for x in ref_likelihood_data]

    # ref_reward_data 데이터 : str -> float
    ref_reward_data = [float(x) for x in ref_reward_data]

    # 분산된 데이터셋 생성
    dist_gen_batch = create_distributed_dataset(strategy, input_seq, target_seq, input_mask, target_mask, ref_likelihood_data, ref_reward_data, my_batch_size)

    '''
    생성 시간 및 진행상황 측정
    '''
    start_time = time.time()
    total_gen_case_num_num = len(train_gen_seq_dir)
    print("\n ref_model : {}, rl_model : {}, \n ref_decoding : {} \n num_gen_case_num : {}/{}".format(ref_model_name, rl_model_name, ref_decoding, gen_case_num + 1, total_gen_case_num_num))
    pb_i = Progbar(None)  # 길이를 지정하지 않음

    '''
    매 gen_case (= epoch) 마다 훈련 정확도, 손실 및 보상 초기화    
    '''
    train_cumul_acc = 0
    train_cumul_loss = 0
    train_cumul_reward = 0

    '''
    루프
    '''
    for idx, dist_inputs in enumerate(dist_gen_batch):

        # 모델 예측 및 업데이트
        train_loss, train_acc, rl_likelihood, dropout_idx = train_step(dist_inputs, rl_model)

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
            pb_i.update(idx + 1, values=metric_values)
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
            pb_i.update(idx + 1, values=metric_values)

        # 배치별 정확도, 손실 및 보상 누계
        train_cumul_acc += train_acc.numpy()
        train_cumul_loss += train_loss.numpy()
        train_cumul_reward += train_reward.numpy()

    # 전체 평균 정확도, 손실 및 보상 (훈련셋)
    train_mean_acc = train_cumul_acc / (idx + 1)
    train_mean_loss = train_cumul_loss / (idx + 1)
    train_mean_reward = train_cumul_reward / (idx + 1)

    # 훈련 손실 히스토리 저장
    train_loss_history.append(train_mean_loss)
    loss_history_pd = pd.DataFrame(train_loss_history, columns=['train_loss'])
    loss_history_pd.to_csv(SAVE_RESULT_DIR + '/loss_history.csv', index_label='epoch')

    # 훈련 정확도 히스토리 저장
    train_acc_history.append(train_mean_acc)
    acc_history_pd = pd.DataFrame(train_acc_history, columns=['train_acc'])
    acc_history_pd.to_csv(SAVE_RESULT_DIR + '/acc_history.csv', index_label='epoch')

    # 훈련 보상 히스토리 저장
    train_reward_history.append(train_mean_reward)
    reward_history_pd = pd.DataFrame(train_reward_history, columns=['train_reward'])
    reward_history_pd.to_csv(SAVE_RESULT_DIR + '/reward_history.csv', index_label='epoch')

    # 가중치 저장 조건
    '''
    test set에 대해서 이전 epoch에서 집계된 최고 성능치보다 현재 epoch의 성능치가 개선될 경우 저장
    '''
    # 현 정확도가 가장 높았던 이전 정확도보다 개선됐을 경우에만 가중치 저장
    rl_model.save_weights(SAVE_WEIGHT_DIR + '/ref_dec={}_epoch={}_weights.ckpt'.format(ref_decoding, gen_case_num))

    if my_dropout in ['quantile', 'adaptive']:
        '''
        드롭아웃 히트맵
        '''
        fig = plt.figure(figsize=(15, 0.5))
        fig.set_facecolor('white')

        dropout_rl_likelihood = rl_likelihood * dropout_idx
        plt.pcolor(tf.reshape(dropout_rl_likelihood, shape=(1, len(rl_likelihood))))
        plt.colorbar()
        plt.savefig(SAVE_RESULT_DIR + '/dropout_heatmap_{}_epoch.pdf'.format(gen_case_num), format='pdf')
        # plt.show()

    # 시간 계산
    end_time = time.time()
    cur_sec = (end_time - start_time) % 60
    cur_min = ((end_time - start_time) // 60) % 60
    cur_hr = ((end_time - start_time) // 60) // 60
    print("elapsed time : {:.0f} hr, {:.0f} min, {:.2f} sec".format(cur_hr, cur_min, cur_sec))
    total_sec = (end_time - total_start_time) % 60
    total_min = ((end_time - total_start_time) // 60) % 60
    total_hr = ((end_time - total_start_time) // 60) // 60
    print("total elapsed time : {:.0f} hr, {:.0f} min, {:.2f} sec".format(total_hr, total_min, total_sec))