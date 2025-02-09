# %%
import os
from pathlib import Path
import argparse
import json
import copy
import numpy as np
import tensorflow as tf
import torch
import time
import glob
import random
from tqdm import tqdm

import einops
from utils import indice_pad_in_prefix, remove_pad_in_prefix_case, right_pad_after_eos_token, createFolder, truncate_datasize_by_ratio, get_truncated_data
from tensorflow.keras.utils import Progbar
from transformers import AutoTokenizer, TFAutoModelForCausalLM, TFGPTJForCausalLM, TFT5ForConditionalGeneration

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

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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
    if len(all_pred_logits) == 256:
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

    return gathered_mean_prob_vec

'''
GPU 셋업
'''
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
initialize_setting()
seed_everything(47)

'''
파라미터 로드
'''
parser = argparse.ArgumentParser(description='receive the parameters')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--ref_model_name', type=str, default='opt_large')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_gen', type=int, default=10)
parser.add_argument('--decoding', type=str, default='beam')
parser.add_argument('--prefix_ratio', type=float, default=0.15)
parser.add_argument('--seq_total_len', type=int, default=30)
args = parser.parse_args()

my_dataset = args.dataset
ref_model_name = args.ref_model_name
my_batch_size = args.batch_size
my_num_gen = args.num_gen                        # number of generations = number of epoch
my_decoding = args.decoding                      # decodnig strategy
my_prefix_ratio = args.prefix_ratio              # ratio of prefix to use
my_seq_total_len = args.seq_total_len            # length of the sequence (prefix_len + gen_len)
# my_dataset = 'sentiment-0'
# ref_model_name = 'gpt2_large'
# my_batch_size = 64
# my_num_gen = 5
# my_decoding = 'stochastic'
# my_prefix_ratio = 0.15
# my_seq_total_len = 30

'''
데이터 로드 경로 설정
'''
# 현 위치의 부모위치 확인
parent_dir = str(Path(os.getcwd()).parents[0])

# 데이터 로드 경로 설정
prep_data_path = parent_dir + '/prep_data' + '/' + my_dataset.split('-')[0] + '/gpt2_small'     # train_input_x, train_input_mask 등 .npy 파일은 항상 /gpt2_small에서 받음.

'''
행동정책 모델 로드
'''
if ref_model_name == 'gpt2_small':
    # GPT2_small의 토크나이저 로드 및 모델 초기화 & 미세조정 가중치 로드
    '''
    행동 정책 모델 (behavior policy model)
    - 행동정책에는 .resize_token_embeddings()를 안하는게 확실히 맞음 !
    '''
    # 토크나이저 로드 및 모델 초기화
    ref_model_config_dir = parent_dir + '/pretrained_weights' + '/' + ref_model_name
    ref_model_tokenizer_dir = ref_model_config_dir + '/tokenizer_right'
    ref_model_weights_dir = ref_model_config_dir + '/model'
    ref_model_tokenizer = AutoTokenizer.from_pretrained(ref_model_tokenizer_dir)
    ref_model = TFAutoModelForCausalLM.from_pretrained(ref_model_weights_dir)
    # ref_model.resize_token_embeddings(len(ref_model_tokenizer))

elif ref_model_name == 'gpt2_large':

    # GPT2_large의 토크나이저 로드 및 모델 초기화 & 미세조정 가중치 로드
    '''
    행동 정책 모델 (behavior policy model)
    - 행동정책에는 .resize_token_embeddings()를 안하는게 확실히 맞음 !
    '''
    # 토크나이저 로드 및 모델 초기화
    ref_model_config_dir = parent_dir + '/pretrained_weights' + '/' + ref_model_name
    ref_model_tokenizer_dir = ref_model_config_dir + '/tokenizer_right'
    ref_model_weights_dir = ref_model_config_dir + '/model'
    ref_model_tokenizer = AutoTokenizer.from_pretrained(ref_model_tokenizer_dir)
    ref_model = TFAutoModelForCausalLM.from_pretrained(ref_model_weights_dir, from_pt=True)
    # ref_model.resize_token_embeddings(len(ref_model_tokenizer))


elif ref_model_name == 'opt' or ref_model_name == 'opt_large' or ref_model_name == 'xglm' or ref_model_name == 'xglm_large':
    '''
    행동 정책 모델 (behavior policy model)
    - 행동정책에는 .resize_token_embeddings()를 안하는게 확실히 맞음 !
    '''
    # 토크나이저 로드 및 모델 초기화
    ref_model_config_dir = parent_dir + '/pretrained_weights' + '/' + ref_model_name
    ref_model_tokenizer_dir = ref_model_config_dir + '/tokenizer_right'
    ref_model_weights_dir = ref_model_config_dir + '/model'
    ref_model_tokenizer = AutoTokenizer.from_pretrained(ref_model_tokenizer_dir)
    ref_model = TFAutoModelForCausalLM.from_pretrained(ref_model_weights_dir, from_pt=True)
    # ref_model.resize_token_embeddings(len(ref_model_tokenizer))


elif ref_model_name == 'gpt_j':
    '''
    행동 정책 모델 (behavior policy model)
    - 행동정책에는 .resize_token_embeddings()를 안하는게 확실히 맞음 !
    '''
    # 토크나이저 로드 및 모델 초기화
    ref_model_config_dir = parent_dir + '/pretrained_weights' + '/' + ref_model_name
    ref_model_tokenizer_dir = ref_model_config_dir + '/tokenizer_right'
    ref_model_weights_dir = ref_model_config_dir + '/model'
    ref_model_tokenizer = AutoTokenizer.from_pretrained(ref_model_tokenizer_dir)
    ref_model = TFGPTJForCausalLM.from_pretrained(ref_model_weights_dir, from_pt=True)
    # ref_model.resize_token_embeddings(len(ref_model_tokenizer))


elif ref_model_name == 't5' or ref_model_name == 'flan_t5' or ref_model_name == 'flan_ul2':
    '''
    행동 정책 모델 (behavior policy model)
    - 행동정책에는 .resize_token_embeddings()를 안하는게 확실히 맞음 !
    '''
    # 토크나이저 로드 및 모델 초기화
    ref_model_config_dir = parent_dir + '/pretrained_weights' + '/' + ref_model_name
    ref_model_tokenizer_dir = ref_model_config_dir + '/tokenizer_right'           # encoder-decoder 모델은 tokenizer_right 아닐껄..?
    ref_model_weights_dir = ref_model_config_dir + '/model'
    ref_model_tokenizer = AutoTokenizer.from_pretrained(ref_model_tokenizer_dir)
    ref_model = TFT5ForConditionalGeneration.from_pretrained(ref_model_weights_dir, from_pt=True)
    # ref_model.resize_token_embeddings(len(ref_model_tokenizer))


'''
훈련용 데이터 셋팅
'''
# 훈련용 인풋 시퀀스 및 어텐션 마스크 로드
# train_input_x = np.load(prep_data_path + '/train_input_x.npy')
# train_input_att = np.load(prep_data_path + '/train_input_att.npy')


# 훈련용 인풋 시퀀스 및 어텐션 마스크 로드
# - train_input_x.npy, train_inpux_att.npy 등이 전부 gpt2_tokenizer로 encoding 되어 있으므로
# - gpt2_tokenizer로 먼저 decoding 한 뒤, ref_model_tokenizer로 다시 encoding 해주는 작업 필요
# - 이렇게 해야 모든 모델들이 "동일한 데이터셋"을 관측할 수 있음.
gpt2_small_config_dir = parent_dir + '/pretrained_weights/gpt2_small'
gpt2_small_tokenizer_dir = gpt2_small_config_dir + '/tokenizer_right'
gpt2_small_tokenizer = AutoTokenizer.from_pretrained(gpt2_small_tokenizer_dir)

train_input_x = np.load(prep_data_path + '/train_input_x.npy')
train_input_x_decode = gpt2_small_tokenizer.batch_decode(train_input_x)
ref_model_tokenizer.padding_side = 'left'
train_input_x_tokenized = ref_model_tokenizer(train_input_x_decode, return_tensors='np', padding=True, add_special_tokens=False)

length_diff_ = train_input_x_tokenized['input_ids'].shape[1] - train_input_x.shape[1]   # gpt2_small_tokenizer 와 ref_model_tokenizer 간 토크나이징 방식 차이로 가장 긴 문장에 대해서 다른 길이로 tokenization이 될 수 있음. 따라서 둘 간의 차이를 구하여 만약 그 차가 양수라면, 즉 ref_model_tokenizer가 더 길게 tokenization 했다면, 그 길이차 만큼 앞 (left)에서 제거 (slice)해주어야 함
if length_diff_ > 0:
    train_input_x = train_input_x_tokenized['input_ids'][:, length_diff_:]
    train_input_att = train_input_x_tokenized['attention_mask'][:, length_diff_:]
else:
    train_input_x = train_input_x_tokenized['input_ids']
    train_input_att = train_input_x_tokenized['attention_mask']

# my_prefix_len 정의
my_prefix_len = int(train_input_x.shape[1] * my_prefix_ratio)

# 데이터 서브 샘플링
# - emotion 데이터 일 때는 이 코드 주석처리하고 돌리기 
# - 즉, emotion 데이터의 경우 데이터 양 자체가 적은데다 label class가 여러개임으로 모든 데이터 전부 활용하도록 설정
if my_dataset.split('-')[0] == 'emotion' or my_dataset.split('-')[0] == 'act':
    '''
    emtion & act 데이터 사이즈 : (76053)
    '''
    truncation_idx = truncate_datasize_by_ratio(data=train_input_x, ratio=0.7)
    train_input_x = get_truncated_data(data=train_input_x, truncation_idx=truncation_idx)
    train_input_att = get_truncated_data(data=train_input_att, truncation_idx=truncation_idx)

elif my_dataset.split('-')[0] == 'politeness':
    '''
    politeness 데이터 사이즈 : (1121980, )
    '''
    truncation_idx = truncate_datasize_by_ratio(data=train_input_x, ratio=0.05)
    train_input_x = get_truncated_data(data=train_input_x, truncation_idx=truncation_idx)
    train_input_att = get_truncated_data(data=train_input_att, truncation_idx=truncation_idx)

elif my_dataset.split('-')[0] == 'sentiment':
    '''
    sentiment 데이터 사이즈 : (560000, )
    '''
    truncation_idx = truncate_datasize_by_ratio(data=train_input_x, ratio=0.1)
    train_input_x = get_truncated_data(data=train_input_x, truncation_idx=truncation_idx)
    train_input_att = get_truncated_data(data=train_input_att, truncation_idx=truncation_idx)

elif my_dataset.split('-')[0] == 'topic':
    '''
    topic 데이터 사이즈 : (120000, )
    '''
    truncation_idx = truncate_datasize_by_ratio(data=train_input_x, ratio=0.5)
    train_input_x = get_truncated_data(data=train_input_x, truncation_idx=truncation_idx)
    train_input_att = get_truncated_data(data=train_input_att, truncation_idx=truncation_idx)

elif my_dataset.split('-')[0] == 'toxicity':
    '''
    toxicity 데이터 사이즈 : (159571, )
    '''
    truncation_idx = truncate_datasize_by_ratio(data=train_input_x, ratio=0.4)
    train_input_x = get_truncated_data(data=train_input_x, truncation_idx=truncation_idx)
    train_input_att = get_truncated_data(data=train_input_att, truncation_idx=truncation_idx)

# '''
# 파라미터 딕셔너리 정의 및 저장
# '''
# kwargs = {
#     'behavior_model' : ref_model_name,
#     'dataset' : my_dataset,
#     'decoding' : my_decoding,
#     'prefix_ratio' : my_prefix_ratio,
#     'prefix_len' : my_prefix_len,
#     'total_len' : my_seq_total_len,
#     'num_generation' : my_num_gen
# }

# SAVE_PARAM_DIR = parent_dir + '/params' + '/' + my_dataset + '/' + ref_model_name
# createFolder(SAVE_PARAM_DIR)
# with open(SAVE_PARAM_DIR + '/kwargs.json', 'w') as f:
#     json.dump(kwargs, f)

'''
주어진 프롬프트 길이 (prefix_len) 내에 <pad>가 존재하는지 여부 확인 및 <pad> 존재 시퀀스 삭제
- 앞서 preprocessing 단계에서 left-padding으로 토크나이징 할 때, 최대 길이를 30으로 제한하였음. 따라서, 길이가 30 미만인 문장들에는 prefix_len 내에 <pad>가 반드시 존재함
- 다시 말해, "prefix_len 내에 <pad>가 존재하는 시퀀스를 삭제한다 = 길이가 30 미만인 문장들 삭제한다" 가 된다.
- 그러나 본질적인 의미는, left-padding에 의해 prefix_len 내에 <pad>가 존재할 경우, <pad>라는 prefix로부터 이어서 문장을 생성하는것은 불가능하므로, 이를 방지하기 위함이다.

참고)
- decoder-only architecture는 <pad>가 앞에서부터 존재하는데,
- reconstruction 포맷으로 (즉, input = output = target (right-shift) 데이터 포맷으로) 학습데이턱 입력되는 경우,
- input의 앞쪽에 등장하는 <pad>가 output (및 target)의 앞쪽에도 동일하게 등장하여,
- 결과적으로 concat(input, output) = input_seq의 중간에 <pad>가 존재하며, 
- decoder-only architecture에 맞지 않는다는 에러를 유발함.
'''
target_idx = indice_pad_in_prefix(prefix_data=train_input_x, prefix_len=my_prefix_len, pad_token_id=ref_model_tokenizer.pad_token_id)
train_input_x = remove_pad_in_prefix_case(target_idx=target_idx, target_data=train_input_x)
train_input_x = tf.cast(train_input_x, dtype=tf.int32)
train_input_att = remove_pad_in_prefix_case(target_idx=target_idx, target_data=train_input_att)
train_input_att = tf.cast(train_input_att, dtype=tf.float32)

'''
다시한번 더 Prefix 자르기
'''
train_input_x_prefixed = train_input_x[:, :my_prefix_len]
train_input_att_prefixed = train_input_att[:, :my_prefix_len]
# train_input_x_prefixed = train_input_x[:64, :my_prefix_len]
# train_input_att_prefixed = train_input_att[:64, :my_prefix_len]

'''
생성용 데이터 셋 구축
'''
with tf.device("/cpu:0"):

    # 데이터셋
    train_dat = tf.data.Dataset.from_tensor_slices((train_input_x_prefixed, 
                                                    train_input_att_prefixed)).shuffle(buffer_size=train_input_x_prefixed.shape[0],
                                                                            reshuffle_each_iteration=False)

    train_batch = train_dat.batch(batch_size=my_batch_size, drop_remainder=True)

'''
생성결과 저장 경로 지정
'''
GEN_SAVE_DIR = parent_dir + '/prep_data' + '/' + my_dataset + '/' + ref_model_name
createFolder(GEN_SAVE_DIR)
print('GEN_SAVE_DIR :', GEN_SAVE_DIR)


'''
모델 얼리기
'''
ref_model.trainable = False

'''
Trajectory Collecting 수행
'''
# 생성 루프
total_start_time = time.time()
full_bp_gen_seq = []
for gen_case_num in range(my_num_gen):

    '''
    [주석처리 하세요.] gen_case_num=0번까지만 확인하는 코드.
    '''
    # if gen_case_num > 0:
    #     break;

    # 생성 시간 및 진행상황 측정
    start_time = time.time()
    print("\n decoding : {} \n gen_case_num : {}/{}".format(my_decoding, gen_case_num + 1, my_num_gen))
    pb_i = Progbar(len(train_batch))

    # 생성 과정
    for idx, (train_input_x_prefixed, train_input_att_prefixed) in enumerate(train_batch):        

        '''
        Trajectory 수집
        - 디코딩 방법 별로 수집
        '''
        if my_decoding == 'stochastic':
            bp_outputs = ref_model.generate(train_input_x_prefixed, attention_mask=train_input_att_prefixed, 
                                                max_length=my_seq_total_len, 
                                                pad_token_id=ref_model_tokenizer.pad_token_id,
                                                eos_token_id=ref_model_tokenizer.eos_token_id,
                                                repetition_penalty=1.2, 
                                                do_sample=True, temperature=1.0,
                                                return_dict_in_generate=True, output_scores=True)
            bp_gen_texts = bp_outputs.sequences
            bp_all_logits = bp_outputs.scores


        elif my_decoding == 'top-k':
            bp_outputs = ref_model.generate(train_input_x_prefixed, attention_mask=train_input_att_prefixed, 
                                                max_length=my_seq_total_len, 
                                                pad_token_id=ref_model_tokenizer.pad_token_id,
                                                eos_token_id=ref_model_tokenizer.eos_token_id,
                                                repetition_penalty=1.2, 
                                                do_sample=True, top_k=10, temperature=1.0,
                                                return_dict_in_generate=True, output_scores=True)
            bp_gen_texts = bp_outputs.sequences
            bp_all_logits = bp_outputs.scores

        elif my_decoding == 'top-p':
            bp_outputs = ref_model.generate(train_input_x_prefixed, attention_mask=train_input_att_prefixed, 
                                                max_length=my_seq_total_len, 
                                                pad_token_id=ref_model_tokenizer.pad_token_id,
                                                eos_token_id=ref_model_tokenizer.eos_token_id,
                                                repetition_penalty=1.2, 
                                                do_sample=True, top_p=0.95, top_k=0, temperature=1.0,
                                                return_dict_in_generate=True, output_scores=True)
            bp_gen_texts = bp_outputs.sequences
            bp_all_logits = bp_outputs.scores


        elif my_decoding == 'beam':
            bp_outputs = ref_model.generate(train_input_x_prefixed, attention_mask=train_input_att_prefixed, 
                                                max_length=my_seq_total_len, 
                                                pad_token_id=ref_model_tokenizer.pad_token_id,
                                                eos_token_id=ref_model_tokenizer.eos_token_id,
                                                num_beams=3, repetition_penalty=1.2, early_stopping=True,
                                                return_dict_in_generate=True, output_scores=True)
            bp_gen_texts = bp_outputs.sequences
            bp_all_logits = bp_outputs.scores


        # 마스킹
        bp_padded_gen_texts = right_pad_after_eos_token(
                                    bp_gen_texts, 
                                    eos_token_id=ref_model_tokenizer.eos_token_id, 
                                    pad_token_id=ref_model_tokenizer.pad_token_id,
                                    total_len = my_seq_total_len
                                    )


        # Likelihood 계산
        my_gen_len = my_seq_total_len - my_prefix_len
        likelihoods = get_likelihood(bp_padded_gen_texts, bp_all_logits, my_prefix_len, my_gen_len)
        likelihood_list = list(tf.cast(likelihoods, dtype=tf.float16).numpy())

        '''
        아래 코드는 나중에 학습 코드에서 적용해줘도 됨.
        '''
        # # 생성된 시퀀스로부터 target_policy_model을 훈련시킬 input, output, target (= right-shifted output) 데이터 만들기
        # bp_gen_input_seq = bp_padded_gen_texts[:, :-1]            # bp_padded_gen_texts[:, :-1] : bp_gen_texts[:, input_len:-1] 가 right_pad 된 상태
        # bp_gen_target_seq = bp_padded_gen_texts[:, 1:]            # bp_padded_gen_texts[:, 1:] : bp_gen_texts[:, input_len+1:] 가 right_pad 된 상태

        # # attention mask 데이터 만들기
        # bp_gen_input_mask = tf.math.not_equal(bp_gen_input_seq, ref_model_tokenizer.pad_token_id)
        # bp_gen_target_mask = tf.math.not_equal(bp_gen_target_seq, ref_model_tokenizer.pad_token_id)

        # 각 배치 (= idx) 에 대해서 통합하기
        if idx == 0:
            total_likelihood_list = copy.deepcopy(likelihood_list)
            total_bp_gen_seq = copy.deepcopy(ref_model_tokenizer.batch_decode(bp_padded_gen_texts))

        else:
            # total_likelihoods = tf.concat([total_likelihoods, likelihoods], axis=0)
            total_likelihood_list += likelihood_list
            total_bp_gen_seq = total_bp_gen_seq + ref_model_tokenizer.batch_decode(bp_padded_gen_texts)

        # 진행상태 바 (Progress Bar) 업데이트
        pb_i.update(idx+1)

    end_time = time.time()
    cur_sec = (end_time - start_time)%60
    cur_min = ((end_time - start_time)//60)%60
    cur_hr = ((end_time - start_time)//60)//60
    print("elapsed time : {:.0f} hr, {:.0f} min, {:.2f} sec".format(cur_hr, cur_min, cur_sec))
    total_sec = (end_time - total_start_time)%60
    total_min = ((end_time - total_start_time)//60)%60
    total_hr = ((end_time - total_start_time)//60)//60
    print("total elapsed time : {:.0f} hr, {:.0f} min, {:.2f} sec".format(total_hr, total_min, total_sec))

    # .txt.포맷으로 저장하기 위해 "list of floats -> list of string" 변환하기
    converted_total_likelihood_list = ['{:.3f}'.format(x) for x in total_likelihood_list]

    # 생성 케이스 별 예측된 우도들 텍스트 파일로 저장하기
    likelihood_txt_file_path = GEN_SAVE_DIR + '/train_likelihood_{}_{}.txt'.format(my_decoding, gen_case_num)
    with open(likelihood_txt_file_path, 'a') as file:
        file.write('\n'.join(converted_total_likelihood_list))

    # gen_input_seq를 .txt 파일로 저장
    train_txt_file_path = GEN_SAVE_DIR + '/train_gen_seq_{}_{}.txt'.format(my_decoding, gen_case_num)
    total_bp_gen_seq = [x.replace('\n\n', '\n').replace('\n', ' ') for x in total_bp_gen_seq]
    with open(train_txt_file_path, 'a') as file:        # 'a' (= add) 인자를 통해 연이어서 데이터를 .txt 파일에 추가하기
        file.write('\n'.join(total_bp_gen_seq))         # '\n' 기준으로 줄바꿈

    # 각 생성 (= gen_case_num) 케이스에 대해서 통합하기
    full_bp_gen_seq += total_bp_gen_seq


# full_bp_gen_seq 를 .txt 파일로 저장
train_txt_file_path = GEN_SAVE_DIR + '/train_gen_seq_{}_full.txt'.format(my_decoding)
with open(train_txt_file_path, 'a') as file:        # 'a' (= add) 인자를 통해 연이어서 데이터를 .txt 파일에 추가하기
    file.write('\n'.join(full_bp_gen_seq))         # <pad> 기준으로 줄바꿈