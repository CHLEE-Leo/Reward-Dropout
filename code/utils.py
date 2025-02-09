# %%
from scipy.special import beta, gamma
import copy
import os
from pathlib import Path
import sys
import argparse
import copy
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

'''
경로 설정
'''
parent_dir = str(Path(os.getcwd()).parents[0])

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

def get_params():
    parser = argparse.ArgumentParser(description='receive the parameters')

    # 필수 인자들 설정
    parser.add_argument('--my_seed', type = int, required = True)
    parser.add_argument('--model', type = str, required = True)         # model : {'gpt2_small', 'gpt2_large', 'dialoGPT', 'bert', 'opt', 'xglm', 'ctrl', 'gpt_j'}
    parser.add_argument('--task', type = str, required = True)          # task : {'ft', 'rl', 'train_eval', 'test_eval'} , ft = fine-tuning, rl = reinforcement-learning
    parser.add_argument('--dataset', type = str, required = True)       # dataset = {'sentiment-0', 'sentiment-1', 
                                                                                    # 'toxicity-0', 'toxicity-1', 
                                                                                    # 'politeness-0', 'politeness-1',
                                                                                    # 'emotion-0', 'emotion-1', 'emotion-2', 'emotion-3', 'emotion-4', 'emotion-5', 'emotion-6',
                                                                                    # 'act-0', 'act-1', 'act-2', 'act-3', 'act-4'}

    parser.add_argument('--batch_size', type = int, required = False)
    parser.add_argument('--lr', type = float, required = False)
    parser.add_argument('--num_epoch', type = int, required = False)
    parser.add_argument('--num_patience', type = int, required = False)

    parser.add_argument('--rl_batch_size', type = int, required = False)        # rl_batch_size : 임의의 정수
    parser.add_argument('--rl_lr', type = float, required = False)              # rl_lr : 임의의 실수
    parser.add_argument('--rl_num_epoch', type = int, required = False)         # rl_num_epoch : 임의의 정수
    parser.add_argument('--decoding', type = str, required = False)             # decoding : {'greedy', 'stochastic', 'top-k', 'top-p', 'beam'}
    parser.add_argument('--prefix_len', type = int, required = False)           # prefix_len : 임의의 정수
    parser.add_argument('--gen_len', type = int, required = False)              # gen_len : 임의의 정수
    parser.add_argument('--dropout', type = str, required = False)              # dropout : 드롭아웃 스타일 {'None', 'random', 'quantile', 'pareto'}
    parser.add_argument('--dropout_rate', type = float, required = False)       # dropout_rate : 드롭아웃 비율, 임의의 [0, 1]
    parser.add_argument('--reward', type = str, required = False)               # reward : 보상 (r), 보상 + 로그 베타 (r+log_b), 보상 + 로그 베타 + 로그 파이 (r+log_b+log_p)


    parser.add_argument('--init_weight', type = str, required = False)          # init_weight : my_model = gpt2_small_init_weight 일 때만 사용.
                                                                                # {'uniform', 'normal', 'Glorot', 'He', 'constant'} 중 하나 입력

    parser.add_argument('--test_prefix', type=str, required=False)              # test_prefix : evaluate_LLM 단계에서 초기 값으로 넣어주는 단어
    global args
    args = parser.parse_args()

    return args

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

'''
폴더 생성
'''
def createFolder(directory: str):
    try:
        if os.path.exists(directory) == False:
            os.makedirs(directory)
            print('\nCreated ! : {}'.format(directory))
        else:
            print('\n"{}" exists yet.'.format(directory))

    except OSError:
        print ('\nError: Creating directory. ' +  directory)


'''
경로 지정 및 생성
'''
def set_SavePath(kwargs, save_file = 'weights'):

    # save_file 및 save_model에 따라 파일저장 경로 별도로 생성
    save_task = '/' + kwargs['task']
    save_model = '/' + kwargs['model_name']
    SAVE_PATH = parent_dir + '/' + save_file + save_task + save_model

    # SAVE_PATH 경로가 존재하지 않는 경우에 해당 경로를 생성
    createFolder(SAVE_PATH)

    return SAVE_PATH

def set_save_dir(kwargs: dict, folder=None, subfolder=None) -> str:

    save_dir = parent_dir + '' + '/' + folder + '/' + subfolder
    save_file_name = name_save_file(kwargs)
    save_file_dir = save_dir + '/' +  save_file_name
    createFolder(save_file_dir)

    return save_file_dir


def name_save_file(kwargs: dict) -> str:
    return ''.join([ '_' + str(val) if idx > 0 else str(val) for idx, val in enumerate(kwargs.values())])


'''
agnews 데이터 csv 파일로 저장
'''
def agnews_to_csv(data, split:str):
    '''
    data : load_data('ag_news')로 내려받은 파일의 .data 
    split : 'train' 또는 'test' ('val'이나 'dev'가 있을 수도 있음)
    '''

    # 데이터 변환
    data_df = pd.DataFrame(list(data[split]))
    data_df = data_df.T
    data_df.columns = ['text', 'label']

    # 저장
    data_df.to_csv(parent_dir + '/data/topic/{}.csv'.format(split))

    return data_df

'''
daily dailog 데이터 전처리 및 저장
'''
def preprocess_n_save_DailyDialog(daily_dialog_raw_dataset, split: str, N_turn: int = None):
    '''
    daily_dialog_raw_dataset : load_dataset('daily_dialog')
    - sentence_dialog : n개의 sentence 묶음인 dialog 단위의 데이터를 sentence로 분해한 데이터
    - dialog_idx : 각 sentence가 어떤 dialog에 속하는지 나타내는 인덱스
    - sentence_dialog_with_index : 각 sentence가 어떤 dialog에 속하는지 인덱스를 달아놓은 데이터
    - full_dataset : dialog, action, emotion 관련 column이 모두 존재하는 데이터셋 
    - cum_sentence_dialog : 각 인덱스 별, 즉 각 dialog 내에서 발생한 sentence들을 순서대로 누적시킨 데이터
    - indexBag : ㅇㅇㅇㅇ
    '''

    dialog_set = daily_dialog_raw_dataset.data[split]['dialog']

    # 대화를 문장 단위로 쪼개어 데이터셋을 구성하고, 대화를 그릅화하는 인덱스 생성
    sentence_dialog = pd.DataFrame([sentence for dialog in dialog_set for sentence in dialog ], columns = ['query'])
    dialog_idx = pd.DataFrame(np.concatenate([np.repeat(idx, len(dialog)) for idx, dialog in enumerate(dialog_set)]), columns=['dialog_idx'])
    sentence_dialog_with_index = pd.concat([dialog_idx, sentence_dialog], axis = 1)

    # 문장단위로 쪼개진 데이터셋에 맞추어 act 및 emotion 컬럼을 구축하고, 하나의 데이터 프레임 full_dataset 으로 정의
    act_list_set = daily_dialog_raw_dataset.data[split]['act']
    act_vec = pd.DataFrame([act for act_list in act_list_set for act in act_list ], columns = ['act_q'])
    full_dataset = pd.concat([sentence_dialog_with_index, act_vec], axis = 1)

    emo_list_set = daily_dialog_raw_dataset.data[split]['emotion']
    emo_vec = pd.DataFrame([emotion for emo_list in emo_list_set for emotion in emo_list ], columns = ['emo_q'])
    full_dataset = pd.concat([full_dataset, emo_vec], axis = 1)

    # 대화의 누적응답 기록 생성
    cum_sentence_dialog, indexBag, all_responseBag, all_act_r_Bag, all_emo_r_Bag = cumulate_sentence_dialog(full_dataset)
    cum_sentence_dialog = cum_sentence_dialog['dialog_history'].apply(lambda x : '<\s>'.join(x))

    # 누적응답 (dialog_history), 질의 (query), 응답 (response) triplet에 맞춘 act_q, emo_q, act_r, emo_r 컬럼을 구축하고, 하나의 데이터 프레임 indiced_full_dataset 으로 정의
    indiced_full_dataset = full_dataset.iloc[indexBag]
    indiced_full_dataset.insert(loc=1, value=list(cum_sentence_dialog), column='dialog_hist')
    indiced_full_dataset.insert(loc=3, value=list(all_responseBag), column='response')
    indiced_full_dataset = pd.concat([indiced_full_dataset, all_act_r_Bag.set_index(indiced_full_dataset.index)], axis = 1)
    indiced_full_dataset = pd.concat([indiced_full_dataset, all_emo_r_Bag.set_index(indiced_full_dataset.index)], axis = 1)

    # 대화 Turn의 갯수가 N_turn개 이하인 대화로만 구성하기
    if N_turn != None:
        N_turn_dialog_idx = np.where(np.unique(indiced_full_dataset['dialog_idx'], return_counts=True)[1] <= N_turn)[0]
        indiced_full_dataset = indiced_full_dataset[indiced_full_dataset['dialog_idx'].isin(N_turn_dialog_idx) == 1]

    # 최종 데이터셋 (indiced_full_dataset) 저장
    indiced_full_dataset.to_csv(parent_dir + '/data/dialog/' + split + '_{}.csv'.format(N_turn))

    # 대화의 평균길이
    mean_dialog_len = np.mean(indiced_full_dataset.groupby('dialog_idx').apply(lambda x : len(x)+1))
    print('{}_dialog mean_len : {}'.format(split, mean_dialog_len))

    return sentence_dialog_with_index, full_dataset, cum_sentence_dialog, indiced_full_dataset

'''
대화기록 누적 함수
'''
def cumulate_sentence_dialog(full_dataset):
    '''
    num_dialog_set : 총 dialog_set 갯수
    historyBag : 각 dialog 마다 진행되는 sentence를 누적시켜 담아놓은 bag 
    '''
    num_dialog_set = len(np.unique(full_dataset['dialog_idx']))
    historyBag = []
    indexBag = []
    all_responseBag = []
    all_act_r_Bag = []
    all_emo_r_Bag = []
    for dialog_idx in range(num_dialog_set):
        a_dialog = full_dataset['query'][full_dataset['dialog_idx'] == dialog_idx]

        queryBag = []
        cum_queryBag = []
        responseBag = []

        # 대화별 누적응답 기록 (responseBag) 추출
        dialog_len = len(a_dialog)
        for idx in range(dialog_len-1):
            query_val = a_dialog.iloc[idx]
            queryBag += [str(query_val)]
            cum_queryBag += [queryBag[:idx+1]]

            response_val = a_dialog.iloc[idx+1]
            responseBag += [str(response_val)]

        # response 및 모든 대화 누적응답 기록 (all_respnoseBag) 추출
        all_responseBag += responseBag
        historyBag += cum_queryBag

        # historyBag 기준으로 full_dataset을 인덱싱하기 위해 인덱스 벡터 추출
        all_idx = list(a_dialog[:idx+1].index)
        indexBag += all_idx
        
        # full_dataset에 달려있는 act / emotion 컬럼은 query 기준이므로, response에 대한 act와 emotion 벡터 별도로 추출
        response_idx = list(a_dialog[1:idx+2].index)
        all_act_r_Bag += list(full_dataset['act_q'].iloc[response_idx])
        all_emo_r_Bag += list(full_dataset['emo_q'].iloc[response_idx])
   
    return pd.DataFrame({'dialog_history' : historyBag}), indexBag, all_responseBag, pd.DataFrame({'act_r' : all_act_r_Bag}), pd.DataFrame({'emo_r' : all_emo_r_Bag})


'''
첫 문장 추출 함수
'''
import re
def extract_first_sentence(paragraph):
    # Use regular expressions to split the paragraph into sentences
    # sentences = re.split(r'(?<=[.!?;:()\[\]{}<>"''“”‘’«»„“‹›«»‹›‟”ˮ‟‘’])\s+|\n|\t', paragraph)
    sentences = re.split(r'(?<=[.!?()\[\]{}<>"''“”‘’«»„“‹›«»‹›‟”ˮ‟‘’])\s+|\n|\t', paragraph)
    
    # Extract the first sentence (remove leading/trailing whitespace)
    first_sentence = sentences[0].strip()
    
    return first_sentence

def extract_n_sentences(paragraph, n):
    # Use regular expressions to split the paragraph into sentences
    # sentences = re.split(r'(?<=[.!?;:()\[\]{}<>"''“”‘’«»„“‹›«»‹›‟”ˮ‟‘’])\s+|\n|\t', paragraph)
    sentences = re.split(r'(?<=[.!?()\[\]{}<>"''“”‘’«»„“‹›«»‹›‟”ˮ‟‘’])\s+|\n|\t', paragraph)
    
    # Extract the first n sentences (remove leading/trailing whitespace)
    extracted_sentences = [sentence.strip() for sentence in sentences[:n]]

    # Get full sentence
    full_sentence = ""
    for extract_sentence in extracted_sentences:
        full_sentence += str(extract_sentence)

    return full_sentence

def get_first_n_words(sentence, n):
    words = sentence.split()
    if n <= len(words):
        return ' '.join(words[:n])
    else:
        return sentence


'''
synthetic_data.py 파일에 사용될 분석용 함수코드
'''
from scipy.stats import norm, truncnorm, truncexpon, poisson

def get_truncated_normal(mean=0, std=1, low=1, upp=10):
    '''
    truncated 정규분포 정의
    '''
    return truncnorm(
        (low - mean) / std, (upp - mean) / std, loc=mean, scale=std)

def get_truncated_poisson(num_sample, mean=None):

    trunc_sample = np.zeros([1, 0])

    while True:

        # 람다 (lambda)가 1인 포아송 분포로부터 num_sample만큼 샘플링
        poisson_tmp_sample = poisson.rvs(mu=mean, size=num_sample)
        trunc_sample = np.append(trunc_sample, poisson_tmp_sample)

        # 1 ~ 10 사이만 필터링
        trunc_sample = trunc_sample[trunc_sample>=1]    # 샘플링 값이 1 이상인 경우만 담기
        trunc_sample = trunc_sample[trunc_sample<=10]   # 샘플링 값이 10 이하인 경우만 담기

        # # 좌측으로 치우쳐진 포아송 분포
        # if side == 'left':

        #     # 람다 (lambda)가 1인 포아송 분포로부터 num_sample만큼 샘플링
        #     poisson_tmp_sample = poisson.rvs(mu=mean, size=num_sample)
        #     trunc_sample = np.append(trunc_sample, poisson_tmp_sample)

        #     # 1 ~ 10 사이만 필터링
        #     trunc_sample = trunc_sample[trunc_sample>=1]    # 샘플링 값이 1 이상인 경우만 담기
        #     trunc_sample = trunc_sample[trunc_sample<=10]   # 샘플링 값이 10 이하인 경우만 담기

        # # 우측으로 치우쳐진 포아송 분포
        # elif side == 'right':

        #     # 람다 (lambda)가 20인 포아송 분포로부터 num_sample만큼 샘플링
        #     # 원래 의도는 람다가 10인 분포를 뽑으려고 한건데, 포아송 분포 특성상 10으로 하니 람다=1일 때와 대칭적인 모습이 나오지 않음.
        #     # 따라서, 람다=20으로 해서 더 right-skewed 만들어 람다=1일 때와 대칭적인 모습이 보이도록 강제.
        #     poisson_tmp_sample = poisson.rvs(mu=20, size=num_sample)
        #     trunc_sample = np.append(trunc_sample, poisson_tmp_sample)

        #     # 1 ~ 10 사이만 필터링
        #     trunc_sample = trunc_sample[trunc_sample>=1]    # 샘플링 값이 1 이상인 경우만 담기
        #     trunc_sample = trunc_sample[trunc_sample<=10]   # 샘플링 값이 10 이하인 경우만 담기

        print('leN_turnc_sample : {}'.format(len(trunc_sample)))
        if len(trunc_sample) >= num_sample:
            return trunc_sample[:num_sample]

def get_samples(num_sample, mu=None, sigma=None, dist='truncnorm', sample_ratio=None, mu1=None, mu2=None, sigma1=None, sigma2=None):

    # x축이 특정 구간으로 절삭된 정규분포 정의
    if dist == 'truncnorm':

        # truncated 정규분포 정의
        trunc_normal_dist = get_truncated_normal(mean=mu, std=sigma, low=1, upp=10)

        # truncated 정규분포로부터 행동 샘플링
        samples = trunc_normal_dist.rvs(num_sample)
        integer_samples = np.round(samples).astype('int32') # 행동 (action)은 이산변수이니 integer로 변환

        # 행동 별 발생빈도 추출
        actions, counts = np.unique(integer_samples, return_counts=True)

    # x축이 특정 구간으로 절삭된 포아송 분포 정의
    elif dist == 'truncpoisson':
        samples = get_truncated_poisson(num_sample, mean=mu)
        integer_samples = samples.astype('int32')
        actions, counts = np.unique(integer_samples, return_counts=True)

    # x축이 특정 구간으로 절삭된 양봉 정규분포 정의
    elif dist == 'trunc_doublenorm':

        # 양봉 truncated 정규분포 정의
        # 두번째 모드의 행동정책 분포의 평균이 첫번째 행동정책 분포의 평균보다 4 더 크게 설정
        trunc_normal_dist1 = get_truncated_normal(mean=mu1, std=sigma1, low=1, upp=10)
        trunc_normal_dist2 = get_truncated_normal(mean=mu2, std=sigma2, low=1, upp=10)

        # 양봉 truncated 정규분포로부터 행동 샘플링
        # 첫번째 모드의 행동정책 분포의 샘플 비율이 전체 샘플의 8할, 두번째 모드가 2할을 포함하도록 설정
        sample1_ratio = sample_ratio
        samples1 = trunc_normal_dist1.rvs(int(num_sample * sample1_ratio))
        integer_samples1 = np.round(samples1).astype('int32') # 행동 (action)은 이산변수이니 integer로 변환

        samples2 = trunc_normal_dist2.rvs(num_sample - int(num_sample * sample1_ratio))
        integer_samples2 = np.round(samples2).astype('int32') # 행동 (action)은 이산변수이니 integer로 변환

        # 행동 별 발생빈도 추출
        integer_samples = np.append(integer_samples1, integer_samples2)
        actions, counts = np.unique(integer_samples, return_counts=True)


    return actions, counts, integer_samples


def get_reward_dist(min_action, a=1.0, b=1.0):
    '''
    보상 샘플추출
    - min_action : reward가 반환되는 최소 절삭 action
    - a, b : 베타 분포의 shape 파라미터
    -
    '''

    x = np.arange(1, 11, 1)

    x_new = x[x>=min_action]
    x[x<min_action] = 0
    x[x>=min_action] = x_new

    # xx = copy.deepcopy(x)
    # xx[xx!=0] = 1

    # x = np.linspace(.01, .99, 10)
    x_1 = np.linspace(.01, .99, len(x[x!=0]))

    y = (1 / beta(a, b)) * x_1 ** (a - 1) * (1 - x_1) ** (b - 1)
    y = y/y.sum()

    # if np.min(y) != np.max(y):
    #     y = (y - np.min(y)) /(np.max(y) - np.min(y))
    # else:
    #     y = 0.5 * np.ones(shape=len(x_1))

    # 0 붙이기
    y_0 = np.zeros(shape = len(x[x==0]))
    reward_dist = np.append(y_0, y)

    # reward_dist = xx * y

    return reward_dist

# def reward_function(x, order=2):

#     # 차수
#     rewards = x**order

#     # # 누적합이 1이 되도록 소프트맥스 정규화
#     # softmax_rewards = np.exp(rewards)/np.exp(rewards).sum()
#     # softmax_rewards = np.exp(rewards)
#     rewards = rewards/rewards.sum()

#     # reward의 값의 범위가 [0, 1]이 되도록 min-max 정규화
#     rewards = (rewards - np.min(rewards)) /(np.max(rewards) - np.min(rewards))

#     return rewards

# def get_reward_dist(min_action, order):
#     '''
#     보상 샘플추출
#     - min_action : reward가 반환되는 최소 절삭 action
#     - order : reward 함수의 차수
#     '''

#     x = np.arange(1, 11, 1)

#     try:
#         x_new = x[x>=min_action]
#         x[x<min_action] = 0
#         x[x>=min_action] = x_new

#     except:
#         pass;

#     # 보상계산
#     reward_dist = reward_function(x, order=order)

#     return reward_dist

def get_rewards(onehot_action, reward_dist):
    
    # reward = tf.reduce_sum(tf.multiply(reward_dist, onehot_action))
    reward = tf.matmul(onehot_action, reward_dist[:, tf.newaxis])

    return reward


def get_synthetic_triplets_by_case(synthetic_data, reward_dist, kwargs):

    # 파라미터 설정
    action_size = kwargs['action_size']
    epi_len = kwargs['epi_len']
    num_cases = kwargs['num_cases']

    # 누적 에피소딕 상태표현의 synthetic dataset 구축
    sampled_actions_data = synthetic_data['bp_sampled_actions'] - 1
    synthetic_onehot_states = tf.one_hot(sampled_actions_data, depth = action_size)                                         # synthetic_onehot_states: (total_sample_size, action_size)
    synthetic_onehot_episodic_states = tf.reshape(synthetic_onehot_states, shape = (num_cases, -1, epi_len, action_size))             # synthetic_onehot_episodic_states : (num_cases, per_case_num_epi, epi_len, action_size)
    synthetic_cum_episodic_states = tf.cumsum(synthetic_onehot_episodic_states, axis = 2)                                       # 에피소드 길이동안 visit frequency 누적

    # 초기 상태 (init_state) 벡터 고려 (epi_len + 1 해주기)
    shape_vec = tf.shape(synthetic_cum_episodic_states).numpy()     # synthetic_cum_episodic_states 형상벡터 추출
    shape_vec[2] = 1                                                # episode축의 차원을 1차원으로 설정
    init_state = tf.zeros(shape = shape_vec)                        # episode의 초기 상태 벡터 정의
    synthetic_cum_episodic_states = tf.concat([init_state, synthetic_cum_episodic_states], axis=2)      # episode축을 따라 초기 상태 벡터 prepend
                                                                                                        # (num_cases, per_case_num_epi, epi_len + 1, action_size)

    # input data (= state_history) 구축
    cur_state = synthetic_cum_episodic_states[:, :, :-1, :]                             # state_history : (num_cases, per_case_num_epi, epi_len + 1, action_size)
                                                                                        # episode의 마지막 state에서는 action sampling이 아니라 종료가 수행됨.
                                                                                        # 즉, 마지막 state에 해당하는 부분은 학습데이터로 고려해주지 않아됨.
                                                                                        # 따라서, 마지막 state를 뺀 나머지 부분들에 대해서 cur_state 정의
                                                                                        # cur_state : (num_cases, per_case_num_epi, epi_len, action_size)
    input_data = copy.deepcopy(cur_state)                                               # input_data : (num_cases, per_case_num_epi, epi_len, action_size)

    # target data (= action history)구축
    target_data = tf.reshape(sampled_actions_data, shape = (num_cases, -1, epi_len, 1))       # target_data : (num_cases, per_case_num_epi, epi_len, 1)

    # reward data (= reward history) 구축
    next_state = copy.deepcopy(synthetic_onehot_episodic_states)        # reward는 next_state에 의해 결정되며, next_state는 action에 의해 정의됨
                                                                        # next_state : (num_cases, per_case_num_epi, epi_len, action_size)
    reward_data = get_rewards(next_state, reward_dist)

    # case 별 분리
    # num_cases = synthetic_cum_episodic_states.shape[0]
    input_data_by_case = [input_data[i, :, :, :] for i in range(num_cases)]        # 케이스 별로 인풋 데이터 따로 담기 : num_cases X (per_case_num_epi, epi_len, action)
    target_data_by_case = [target_data[i, :, :, :] for i in range(num_cases)]      # 케이스 별로 타겟 데이터 따로 담기 : num_cases X (per_case_num_epi, epi_len, action)
    reward_data_by_case = [reward_data[i, :, :, :] for i in range(num_cases)]      # 케이스 별로 보상 데이터 따로 담기 : num_cases X (per_case_num_epi, epi_len, action)

    return input_data_by_case, target_data_by_case, reward_data_by_case


'''
데이터 사이즈를 특정 비율로 자르기
'''
def truncate_datasize_by_ratio(data, ratio:float=None):
    '''
    ratio : (0, 1]
    '''

    # 전체 데이터셋에서 주어진 ratio에 해당하는 만큼의 샘플 갯수 정의
    num_samples = int(data.shape[0] * ratio)

    # 전체 데이터셋에 대한 인덱스 정의
    all_idx = tf.range(0, data.shape[0])

    # 전체 데이터셋 셔플링 후 ratio에 해당하는 만큼의 샘플 갯수만 추출
    truncation_idx = tf.random.shuffle(all_idx)[:num_samples]

    # 인덱스 소팅 
    truncation_idx = tf.sort(truncation_idx)

    return truncation_idx

def get_truncated_data(data, truncation_idx):
    trunc_data = tf.gather(params=data, indices=truncation_idx, axis=0)

    return trunc_data

'''
주어진 프롬프트 길이 내애 <pad>가 존재하는 경우 인덱싱 및 제거
'''
def indice_pad_in_prefix(prefix_data, prefix_len:int, pad_token_id:int):

    # 프롬프트 내에 <pad>가 존재하지 않는 경우 hit
    target_idx = tf.math.not_equal(prefix_data[:, :prefix_len], pad_token_id)
    target_idx = tf.cast(target_idx, dtype=tf.int32)

    # 샘플별 hit의 총합이 프롬프트 길이와 같은 경우만 인덱싱
    # 즉, 프롬프트 내에 <pad>가 존재하지 않는 경우 인덱싱
    target_idx = tf.where(tf.reduce_sum(target_idx, axis = 1) == prefix_len)

    return target_idx

def remove_pad_in_prefix_case(target_idx, target_data):

    # target_idx를 활용해 target_data 인덱싱
    indiced_target_data = tf.squeeze(tf.gather(target_data, target_idx))

    return tf.reshape(indiced_target_data, shape = (-1, target_data.shape[1]))

'''
eos_token이 우연이 제일 마지막 토큰으로 또는 아예 안나오는 경우에 대해서
- case 1: Last index -> Last index + 1 -> 1st index
- case 2: None -> 1st index -> Last index
의 두 경우를 순차적으로 수행하여, 전부 Last index 으로 처리
'''
def eos_index_change(gen_inputs, after_eos_token_col_idx, eos_idx_of_interest):

    '''
    --------------------------------------------------------------------------------------------------
    case 1 : eos_idx_of_interest = tf.shape(gen_inputs)[1] + 1
    
    - eos_token_idx : Last = tf.shape(gen_inputs)[1]-1 --(+1)--> tf.shape(gen_inputs)[1] -> 1
    - eos_idx_of_interest : tf.shape(gen_inputs)[1]
    - eos index change : tf.shape(gen_inputs)[1] -> 1

    - summary : Last index -> Last index + 1 -> 1st index
    - return : after_eos_token_col_idx_updated

    --------------------------------------------------------------------------------------------------
    case 2 : eos_idx_of_interest = 1

    - eos_token_idx : None -> 0 --(+1)--> 1 -> tf.shape(gen_inputs)[1]-1
    - eos_idx_of_interest : 1
    - eos index change : 1 -> tf.shape(gen_inputs)[1]-1

    - summary : None -> 1st index -> Last index
    - return : after_eos_token_col_idx_updated
    --------------------------------------------------------------------------------------------------
    '''

    # 생성결과 (gen_inputs)에 eos_idx_of_interest 이 존재하는 경우, 해당 생성결과의 row_idx 뽑기
    target_row_idx = tf.where(tf.math.equal(after_eos_token_col_idx, eos_idx_of_interest))


    # (case 1)의 경우
    # - eos_token이 마지막 시점에 등장한 (after_eos_token_col_idx = tf.shape(gen_inputs).shape[1]+1 인) row_idx들에 대해서,
    # - eos_token이 등장한 col_idx 값을 1st 시점 (= 1) 로 바꿔주기 위해
    # - 1 값을 target_row_idx 갯수만큼 복제
    # - 이 복제된 eos_update_idx 는 after_eos_token_col_idx = tf.shape(gen_inputs).shape[1]+1 인 부분 (= target_row_idx)을
    # - 1 값으로 update
    if eos_idx_of_interest != 1:
        eos_update_idx = tf.repeat(1, tf.shape(target_row_idx)[0])
        after_eos_token_col_idx2 = tf.tensor_scatter_nd_update(tensor=after_eos_token_col_idx, 
                                                        indices=target_row_idx, 
                                                        updates=eos_update_idx)
    # (case 2)의 경우
    # - eos_token이 존재하지 않은 (after_eos_token_col_idx = 1 인) row_idx들에 대해서,
    # - eos_token이 등장한 col_idx 값을 마지막 시점 (= tf.shape(gen_inputs)[1]-1) 로 바꿔주기 위해
    # - tf.shape(gen_inputs)[1]-1 값을 target_row_idx 갯수만큼 복제
    # - 이 복제된 eos_update_idx 는 after_eos_token_col_idx = 1 인 부분 (= target_row_idx)을
    # - tf.shape(gen_inputs)[1]-1 값으로 update
    else:
        eos_update_idx = tf.repeat(tf.shape(gen_inputs)[1]-1, tf.shape(target_row_idx)[0])
        after_eos_token_col_idx2 = tf.tensor_scatter_nd_update(tensor=after_eos_token_col_idx, 
                                                        indices=target_row_idx, 
                                                        updates=eos_update_idx)


    return after_eos_token_col_idx2


'''
eos_token 직후 (aka right-direction)의 모든 토큰들을 패딩
'''
def right_pad_after_eos_token(gen_inputs, eos_token_id:int, pad_token_id:int, total_len:int):

    '''
    모든 샘플에 대해서 eos_token 직후 첫 컬럼위치를 인덱싱 = after_eos_token_col_idx
    '''
    eos_token_hit_matrix = tf.math.equal(gen_inputs, eos_token_id)

    # eos는 포함시키면 안되고, 그 뒤 시점부터 padding의 대상임. 따라서, eos가 등장한 시점인 argmax의 이후 시점을 인식하도록 +1 해주어야 함.
    after_eos_token_col_idx = tf.argmax(eos_token_hit_matrix, axis = -1).numpy() + 1

    '''
    eos_token 위치 업데이트 첫번째 (case 1)
    - eos_token이 Last index에 등장한 샘플의 경우 (= after_eos_token_col_idx = tf.shape(gen_inputs)[1]+1)에 대한 경우 
    - eos_token_idx : Last index -> Last index + 1 -> 1st index
    '''
    after_eos_token_col_idx_updated = eos_index_change(gen_inputs,
                                                       after_eos_token_col_idx = after_eos_token_col_idx,
                                                       eos_idx_of_interest = tf.shape(gen_inputs)[1])

    '''
    eos_token 위치 업데이트 두번째 (case 2)
    - eos_token이 존재하지 않은 샘플의 경우 (= after_eos_token_col_idx = 1)에 대한 경우
    - eos_token_idx : None -> 1st index -> Last index
    '''
    after_eos_token_col_idx_updated = eos_index_change(gen_inputs,
                                                       after_eos_token_col_idx = after_eos_token_col_idx_updated,
                                                       eos_idx_of_interest = 1)


    '''
    최종 eos_token 이후 토큰들의 인덱스 (= after_eos_token_col_idx_updated) 를 가지고 padding 수행
    '''
    # 업데이트 된 after_eos_token_col_idx (= after_eos_token_col_idx2)로 col/row idx 추출
    after_eos_token_col_idx3 = tf.ragged.range(after_eos_token_col_idx_updated, total_len)
    after_eos_token_row_idx = [np.repeat(i, len(val)) for i, val in enumerate(after_eos_token_col_idx3)]

    col_idx = np.concatenate(list(after_eos_token_col_idx3))[:, np.newaxis]
    row_idx = np.concatenate(after_eos_token_row_idx)[:, np.newaxis]
    target_idx = np.concatenate([row_idx, col_idx], axis = -1)

    pad_matrix = tf.ones(shape=tf.shape(target_idx)[0], dtype = tf.int32) * pad_token_id

    gen_inputs_right_pad = tf.tensor_scatter_nd_update(tensor=gen_inputs, indices=target_idx, updates=pad_matrix)

    return gen_inputs_right_pad


def custom_generation(model, initial_data, decoding='greedy', max_gen_len=15):
    gen_seqs = copy.deepcopy(initial_data)

    for i in range(max_gen_len):
        dec_outputs = model(gen_seqs)
    
       # 보통 dec_outputs은 여러 array를 원소로 갖는 list로 정의되는데, 
       # 이 때 첫번째 원소가 대게 logits 값이므로 dec_outputs[0]으로 해주었음

        if decoding == 'greedy':
            top_k = 1
            preds = tf.math.top_k(dec_outputs[0][:, -1, :], k = top_k)[1]          # get a token of maximal probability at each time step
            # preds = tf.expand_dims(preds, axis = -1)                            # get a token of the last time step
            gen_seqs = tf.concat([gen_seqs, tf.cast(preds, dtype=tf.int32)], axis=-1)

        elif decoding == 'stochastic':
            preds = tf.random.categorical(logits = dec_outputs[0][:, -1, :], num_samples = 1)
            gen_seqs = tf.concat([gen_seqs, tf.cast(preds, dtype=tf.int32)], axis=-1)

    return gen_seqs