# %%
'''
필요한 라이브러리 로드
'''
import os
from pathlib import Path
import argparse
import copy
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from utils import *

parser = argparse.ArgumentParser(description='receive the parameters')
parser.add_argument('--dataset', type=str, required = True)       # dataset = {'sentiment-0', 'sentiment-1', 
                                                                              # 'toxicity-0', 'toxicity-1', 
                                                                              # 'politeness-0', 'politeness-1',
                                                                              # 'emotion-0', 'emotion-1', 'emotion-2', 'emotion-3', 'emotion-4', 'emotion-5', 'emotion-6',
                                                                              # 'act-0', 'act-1', 'act-2', 'act-3', 'act-4'}
parser.add_argument('--model', type=str, required=True)             # model = {'gpt2_small', 'gpt2_large', 
                                                                            #  'opt', 'opt_large', 
                                                                            #  'xglm', 'xglm_large', 
                                                                            #  'gpt_j', 
                                                                            #  't5', 'flan_t5', 'flan_ul2', 
                                                                            #  'bert'}
parser.add_argument('--max_len', type=int, required=True)
args = parser.parse_args()
my_dataset = args.dataset
my_model = args.model
my_max_len = args.max_len

'''
경로 설정
'''
parent_dir = str(Path(os.getcwd()).parents[0])

'''
각 데이터 셋을 균일한 포맷으로 전처리 한 뒤, 그것을 다시 'prep_data' dir에 저장

- 모든 데이터들을 토크나이징 한 뒤 train_np - test_np로 나누어서 tensor (.np, .tf, .pt) 포맷 파일로 각 prep_data에 저장
--> 저장 데이터 1 : 시퀀스 데이터
--> 저장 데이터 2 : 라벨 데이터 
----> (1) 긍부정 : negative (0) / postive (1)
----> (2) 공손함 : polite (P9 = 1) / non-polite (P0-P8 = 0)
----> (3) 유해성 : non-toxic (0) / toxic (1) 
----> (4) 감정 : no-emotion (0) / anger (1) / disgust (2) / fear (3) / happiness (4) / sadness (5) / surprise (6)
----> (5) 대화행위 : __dummy__ (0) / inform (1) / question (2) / directive (3) / commissive (4)
----> (6) 자연스러움 :
----> (7) 주제 : World (0), Sports (1), Business (2), Sci/Tech (3)

- 토크나이저는 각 model 별로 (e.g., gpt2_small, gpt2_large, dialoGPT, bert) 계속 불러와서 사용하면 되니 저장할 필요 X
--> 모델 인자를 통해 토크나이저 로드
'''

# (1) 토크나이저 로드
# if 'dialog' in my_dataset:
#     '''
#     dialog 데이터셋은 generative model 훈련용 데이터 이며 tokenizer 선택시 모델로써 여러 버전의 gpt를 고려할 수 있음 
#     my_model = input('input gpt2 version : gpt2_small / gpt2_large / dialoGPT')
#     그러나, tokenizer를 모델별로 따로 구축하면 데이터셋도 모델 갯수만큼 필요하게 되는데 이는 매우 비효율적임.
#     따라서 우리는 모든 대화 시퀀스 데이터는 'gpt2_small'로 토크나이징 하는 것으로 통일하겠음.
#     '''

#     my_model = 'gpt2_small'
#     tokenizer_path = parent_dir + '/pretrained_weights/' + my_model
#     my_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path + '/tokenizer')

# else:
#     '''
#     dialog 데이터셋 외 다른 데이터는 모두 reward model 훈련용 데이터 이므로 tokenizer 선택시 모델은 'bert'로 고정
#     '''
#     my_model = 'bert'
#     tokenizer_path = parent_dir + '/pretrained_weights/' + my_model
#     my_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path + '/tokenizer', truncation = True, padding = 'max_length')

# # (2) 데이터셋 로드 및 정체 후 저장
# if 'dialog' in my_dataset:
#     '''
#     대화 히스토리 - 응답 데이터셋 구축
#     - Conversation AI를 훈련시키기 위한 데이터
#     - N_turn : 각 대화를 구성하는 턴의 횟수 (즉, 대회를 주고 받은 횟수 = 대화 길이)
#     '''
#     N_turn = None

#     # train 데이터의 input/target 시퀀스 로드 및 정제
#     train_data = pd.read_csv(parent_dir + '/data/' + my_dataset + '/train_{}.csv'.format(N_turn), index_col=0)
#     train_input_x = train_data['dialog_hist']
#     train_input_x_dict = left_tokenizer(list(train_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
#     train_input_x_np = train_input_x_dict['input_ids']
#     train_input_x_att_mask_np = train_input_x_dict['attention_mask']

#     train_target_y = train_data['response']
#     train_target_y = train_target_y.apply(lambda x : '<bos>' + x + '<eos>')     # 타겟 시퀀스는 앞뒤로 <bos>와 <eos> 토큰 붙여주기
#     train_target_y_dict = right_tokenizer(list(train_target_y), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)       # 타겟 시퀀스 토크나이징
#     train_target_y_np = train_target_y_dict['input_ids']
#     train_target_y_att_mask_np = train_target_y_dict['attention_mask']

#     # test 데이터의 input/target 시퀀스 로드 및 정제
#     test_data = pd.read_csv(parent_dir + '/data/' + my_dataset + '/test_{}.csv'.format(N_turn), index_col=0)
#     test_input_x = test_data['dialog_hist']
#     test_input_x_dict = left_tokenizer(list(test_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
#     test_input_x_np = test_input_x_dict['input_ids']
#     test_input_x_att_mask_np = test_input_x_dict['attention_mask']

#     test_target_y = test_data['response']
#     test_target_y = test_target_y.apply(lambda x : '<bos>' + x + '<eos>')
#     test_target_y_dict = right_tokenizer(list(test_target_y), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)       # 타겟 시퀀스 토크나이징
#     test_target_y_np = test_target_y_dict['input_ids']
#     test_target_y_att_mask_np = test_target_y_dict['attention_mask']

#     # 저장경로 지정 및 생성
#     save_dir = parent_dir + '/prep_data' + '/' + my_dataset + '/' + my_model
#     createFolder(save_dir)

#     # train/test input_x 시퀀스 데이터 .np 파일로 저장
#     train_np_file_path = save_dir + '/train_input_x.npy'
#     test_np_file_path = save_dir + '/test_input_x.npy'
#     np.save(train_np_file_path, train_input_x_np)
#     np.save(test_np_file_path, test_input_x_np)

#     # train/test input_x 어텐션 데이터 .np 파일로 저장
#     train_np_file_path = save_dir + '/train_input_att.npy'
#     test_np_file_path = save_dir + '/test_input_att.npy'
#     np.save(train_np_file_path, train_input_x_att_mask_np)
#     np.save(test_np_file_path, test_input_x_att_mask_np)

#     # train/test target_y 시퀀스 데이터 .np 파일로 저장
#     train_np_file_path = save_dir + '/train_target_y.npy'
#     test_np_file_path = save_dir + '/test_target_y.npy'
#     np.save(train_np_file_path, train_target_y_np)
#     np.save(test_np_file_path, test_target_y_np)

#     # train/test target_y 어텐션 데이터 .np 파일로 저장
#     train_np_file_path = save_dir + '/train_target_att.npy'
#     test_np_file_path = save_dir + '/test_target_att.npy'
#     np.save(train_np_file_path, train_target_y_att_mask_np)
#     np.save(test_np_file_path, test_target_y_att_mask_np)


if 'sentiment' in my_dataset:
    '''
    긍부정 시퀀스 - 라벨 데이터 구축
    - 긍부정 분류기를 훈련시키기 위한 데이터
    - 긍부정 : negative (0) / postive (1)
    '''

    if my_model == 'bert':

        '''
        토크나이저
        '''
        tokenizer_path = parent_dir + '/pretrained_weights/' + my_model
        my_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path + '/tokenizer', truncation = True, padding = 'max_length')


        '''
        train 데이터
        '''
        # train 데이터의 input/target 시퀀스 로드 및 정제
        train_data = pd.read_csv(parent_dir + '/data/' + my_dataset + '/train.csv', index_col=0)
        train_input_x = train_data['text']
        
        train_input_x_dict = my_tokenizer(list(train_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        train_input_x_np = train_input_x_dict['input_ids']
        train_input_x_att_mask_np = train_input_x_dict['attention_mask']

        train_target_y = train_data['label']
        # train_target_y = train_target_y.iloc[target_idx]
        train_target_y_np = np.array(train_target_y)

        '''
        test 데이터
        '''
        # test 데이터의 input/target 시퀀스 로드 및 정제
        test_data = pd.read_csv(parent_dir + '/data/' + my_dataset + '/test.csv', index_col=0)
        test_input_x = test_data['text']
        # test_input_x, trunc_len, target_idx = truncate_data(test_input_x, quantile=my_quantile)
        test_input_x_dict = my_tokenizer(list(test_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        test_input_x_np = test_input_x_dict['input_ids']
        test_input_x_att_mask_np = test_input_x_dict['attention_mask']

        test_target_y = test_data['label']
        # test_target_y = test_target_y.iloc[target_idx]
        test_target_y_np = np.array(test_target_y)

        '''
        데이터 저장
        '''
        # 저장경로 지정 및 생성
        save_dir = parent_dir + '/prep_data' + '/' + my_dataset + '/' + my_model
        createFolder(save_dir)

        # train/test input_x 시퀀스 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_input_x.npy'
        test_np_file_path = save_dir + '/test_input_x.npy'
        np.save(train_np_file_path, train_input_x_np)
        np.save(test_np_file_path, test_input_x_np)

        # train/test input_x 어텐션 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_input_att.npy'
        test_np_file_path = save_dir + '/test_input_att.npy'
        np.save(train_np_file_path, train_input_x_att_mask_np)
        np.save(test_np_file_path, test_input_x_att_mask_np)

        # train/test target_y 라벨 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_target_y.npy'
        test_np_file_path = save_dir + '/test_target_y.npy'
        np.save(train_np_file_path, train_target_y_np)
        np.save(test_np_file_path, test_target_y_np)

    else:
        '''
        토크나이저
        '''
        tokenizer_path = parent_dir + '/pretrained_weights/' + my_model
        left_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path + '/tokenizer_left')
        right_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path + '/tokenizer_right')

        '''
        train 데이터
        '''
        # train 데이터의 input 시퀀스 로드 및 정제
        train_data = pd.read_csv(parent_dir + '/data/' + my_dataset + '/train.csv', index_col=0)
        train_input_x = train_data['text']
        
        train_input_x_dict = left_tokenizer(list(train_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        train_input_x_np = train_input_x_dict['input_ids']
        train_input_x_att_mask_np = train_input_x_dict['attention_mask']

        # train 데이터의 y 시퀀스 로드 및 정제
        train_y = train_data['text']
        train_y = train_y.apply(lambda y : '<bos>' + y + '<eos>')
        # train_y = train_y.iloc[target_idx]
        train_y_dict = right_tokenizer(list(train_y), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        train_y_np = train_y_dict['input_ids']
        train_y_att_mask_np = train_y_dict['attention_mask']


        '''
        test 데이터
        '''
        # test 데이터의 input 시퀀스 로드 및 정제
        test_data = pd.read_csv(parent_dir + '/data/' + my_dataset + '/test.csv', index_col=0)
        test_input_x = test_data['text']
        # test_input_x, trunc_len, target_idx = truncate_data(test_input_x, quantile=my_quantile)
        test_input_x_dict = left_tokenizer(list(test_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        test_input_x_np = test_input_x_dict['input_ids']
        test_input_x_att_mask_np = test_input_x_dict['attention_mask']

        # test 데이터의 y 시퀀스 로드 및 정제
        test_y = test_data['text']
        test_y = test_y.apply(lambda y : '<bos>' + y + '<eos>')
        # test_y = test_y.iloc[target_idx]
        test_y_dict = right_tokenizer(list(test_y), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        test_y_np = test_y_dict['input_ids']
        test_y_att_mask_np = test_y_dict['attention_mask']


        '''
        데이터 저장
        '''
        # 저장경로 지정 및 생성
        save_dir = parent_dir + '/prep_data' + '/' + my_dataset + '/' + my_model
        createFolder(save_dir)

        # train/test input_x 시퀀스 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_input_x.npy'
        test_np_file_path = save_dir + '/test_input_x.npy'
        np.save(train_np_file_path, train_input_x_np)
        np.save(test_np_file_path, test_input_x_np)

        # train/test input_x 어텐션 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_input_att.npy'
        test_np_file_path = save_dir + '/test_input_att.npy'
        np.save(train_np_file_path, train_input_x_att_mask_np)
        np.save(test_np_file_path, test_input_x_att_mask_np)

        # train/test target_y 시퀀스 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_y.npy'
        test_np_file_path = save_dir + '/test_y.npy'
        np.save(train_np_file_path, train_y_np)
        np.save(test_np_file_path, test_y_np)

        # train/test target_y 어텐션 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_y_att.npy'
        test_np_file_path = save_dir + '/test_y_att.npy'
        np.save(train_np_file_path, train_y_att_mask_np)
        np.save(test_np_file_path, test_y_att_mask_np)


elif 'politeness' in my_dataset:
    '''
    공손함 시퀀스 - 라벨 데이터 구축
    - 공손함 분류기를 훈련시키기 위한 데이터
    - 공손함 : polite (P9 = 1) / non-polite (P0-P8 = 0)
    '''

    if my_model == 'bert':
    
        '''
        토크나이저
        '''
        tokenizer_path = parent_dir + '/pretrained_weights/' + my_model
        my_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path + '/tokenizer', truncation = True, padding = 'max_length')


        '''
        train 데이터
        '''
        # train 데이터의 input/target 시퀀스 로드 및 정제
        train_data = pd.read_csv(parent_dir + '/data/' + my_dataset + '/train.csv', index_col=0)
        train_data.reset_index(inplace=True, drop=True)
        label_one_idx = np.where(train_data['style'] == 'P_9')[0]
        label_zero_idx = np.where(train_data['style'] != 'P_9')[0]
        train_data['style'][list(label_one_idx)] = 1
        train_data['style'][list(label_zero_idx)] = 0

        train_input_x = train_data['txt']
        
        train_input_x_dict = my_tokenizer(list(train_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        train_input_x_np = train_input_x_dict['input_ids']
        train_input_x_att_mask_np = train_input_x_dict['attention_mask']

        train_target_y = train_data['style']
        # train_target_y = train_target_y.iloc[target_idx]
        train_target_y_np = np.array(train_target_y).astype('int64')


        '''
        test 데이터
        '''
        # test 데이터의 input/target 시퀀스 로드 및 정제
        test_data = pd.read_csv(parent_dir + '/data/' + my_dataset + '/test.csv', index_col=0)
        test_data.reset_index(inplace=True, drop=True)
        label_one_idx = np.where(test_data['style'] == 'P_9')[0]
        label_zero_idx = np.where(test_data['style'] != 'P_9')[0]
        test_data['style'][list(label_one_idx)] = 1
        test_data['style'][list(label_zero_idx)] = 0

        test_input_x = test_data['txt']
        # test_input_x, trunc_len, target_idx = truncate_data(test_input_x, quantile=my_quantile)
        test_input_x_dict = my_tokenizer(list(test_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        test_input_x_np = test_input_x_dict['input_ids']
        test_input_x_att_mask_np = test_input_x_dict['attention_mask']

        test_target_y = test_data['style']
        # test_target_y = test_target_y.iloc[target_idx]
        test_target_y_np = np.array(test_target_y).astype('int64')


        '''
        데이터 저장
        '''
        # 저장경로 지정 및 생성
        save_dir = parent_dir + '/prep_data' + '/' + my_dataset + '/' + my_model
        createFolder(save_dir)

        # train/test input_x 시퀀스 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_input_x.npy'
        test_np_file_path = save_dir + '/test_input_x.npy'
        np.save(train_np_file_path, train_input_x_np)
        np.save(test_np_file_path, test_input_x_np)

        # train/test input_x 어텐션 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_input_att.npy'
        test_np_file_path = save_dir + '/test_input_att.npy'
        np.save(train_np_file_path, train_input_x_att_mask_np)
        np.save(test_np_file_path, test_input_x_att_mask_np)

        # train/test target_y 라벨 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_target_y.npy'
        test_np_file_path = save_dir + '/test_target_y.npy'
        np.save(train_np_file_path, train_target_y_np)
        np.save(test_np_file_path, test_target_y_np)

    else:
        '''
        토크나이저
        '''
        tokenizer_path = parent_dir + '/pretrained_weights/' + my_model
        left_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path + '/tokenizer_left')
        right_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path + '/tokenizer_right')

        '''
        train 데이터
        '''
        # train 데이터의 input 시퀀스 로드 및 정제
        train_data = pd.read_csv(parent_dir + '/data/' + my_dataset + '/train.csv', index_col=0)
        train_input_x = train_data['txt']
        
        train_input_x_dict = left_tokenizer(list(train_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        train_input_x_np = train_input_x_dict['input_ids']
        
        if my_model == 'gpt2_small' or my_model == 'gpt2_large' or my_model == 'gpt_j' or my_model == 'opt' or my_model == 'xglm' or my_model == 'ctrl':
            train_input_x_att_mask_np = train_input_x_dict['attention_mask']

        # train 데이터의 y 시퀀스 로드 및 정제
        train_y = train_data['txt']
        train_y = train_y.apply(lambda y : '<bos>' + y + '<eos>')
        # train_y = train_y.iloc[target_idx]
        train_y_dict = right_tokenizer(list(train_y), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        train_y_np = train_y_dict['input_ids']

        if my_model == 'gpt2_small' or my_model == 'gpt2_large' or my_model == 'gpt_j' or my_model == 'opt' or my_model == 'xglm' or my_model == 'ctrl':
            train_y_att_mask_np = train_y_dict['attention_mask']


        '''
        test 데이터
        '''
        # test 데이터의 input 시퀀스 로드 및 정제
        test_data = pd.read_csv(parent_dir + '/data/' + my_dataset + '/test.csv', index_col=0)
        test_input_x = test_data['txt']
        # test_input_x, trunc_len, target_idx = truncate_data(test_input_x, quantile=my_quantile)
        test_input_x_dict = left_tokenizer(list(test_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        test_input_x_np = test_input_x_dict['input_ids']
        
        if my_model == 'gpt2_small' or my_model == 'gpt2_large' or my_model == 'gpt_j' or my_model == 'opt' or my_model == 'xglm' or my_model == 'ctrl':
            test_input_x_att_mask_np = test_input_x_dict['attention_mask']

        # test 데이터의 y 시퀀스 로드 및 정제
        test_y = test_data['txt']
        test_y = test_y.apply(lambda y : '<bos>' + y + '<eos>')
        # test_y = test_y.iloc[target_idx]
        test_y_dict = right_tokenizer(list(test_y), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        test_y_np = test_y_dict['input_ids']

        if my_model == 'gpt2_small' or my_model == 'gpt2_large' or my_model == 'gpt_j' or my_model == 'opt' or my_model == 'xglm' or my_model == 'ctrl':
            test_y_att_mask_np = test_y_dict['attention_mask']


        '''
        데이터 저장
        '''
        # 저장경로 지정 및 생성
        save_dir = parent_dir + '/prep_data' + '/' + my_dataset + '/' + my_model
        createFolder(save_dir)

        # train/test input_x 시퀀스 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_input_x.npy'
        test_np_file_path = save_dir + '/test_input_x.npy'
        np.save(train_np_file_path, train_input_x_np)
        np.save(test_np_file_path, test_input_x_np)

        if my_model == 'gpt2_small' or my_model == 'gpt2_large' or my_model == 'gpt_j' or my_model == 'opt' or my_model == 'xglm' or my_model == 'ctrl':
            # train/test input_x 어텐션 데이터 .np 파일로 저장
            train_np_file_path = save_dir + '/train_input_att.npy'
            test_np_file_path = save_dir + '/test_input_att.npy'
            np.save(train_np_file_path, train_input_x_att_mask_np)
            np.save(test_np_file_path, test_input_x_att_mask_np)

        # train/test target_y 시퀀스 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_y.npy'
        test_np_file_path = save_dir + '/test_y.npy'
        np.save(train_np_file_path, train_y_np)
        np.save(test_np_file_path, test_y_np)

        if my_model == 'gpt2_small' or my_model == 'gpt2_large' or my_model == 'gpt_j' or my_model == 'opt' or my_model == 'xglm' or my_model == 'ctrl':
            # train/test target_y 어텐션 데이터 .np 파일로 저장
            train_np_file_path = save_dir + '/train_y_att.npy'
            test_np_file_path = save_dir + '/test_y_att.npy'
            np.save(train_np_file_path, train_y_att_mask_np)
            np.save(test_np_file_path, test_y_att_mask_np)


if 'topic' in my_dataset:
    '''
    긍부정 시퀀스 - 라벨 데이터 구축
    - 긍부정 분류기를 훈련시키기 위한 데이터
    - 긍부정 : negative (0) / postive (1)
    '''

    if my_model == 'bert':

        '''
        토크나이저
        '''
        tokenizer_path = parent_dir + '/pretrained_weights/' + my_model
        my_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path + '/tokenizer', truncation = True, padding = 'max_length')


        '''
        train 데이터
        '''
        # train 데이터의 input/target 시퀀스 로드 및 정제
        train_data = pd.read_csv(parent_dir + '/data/' + my_dataset + '/train.csv', index_col=0)
        train_input_x = train_data['text']
        
        train_input_x_dict = my_tokenizer(list(train_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        train_input_x_np = train_input_x_dict['input_ids']
        train_input_x_att_mask_np = train_input_x_dict['attention_mask']

        train_target_y = train_data['label']
        # train_target_y = train_target_y.iloc[target_idx]
        train_target_y_np = np.array(train_target_y)

        '''
        test 데이터
        '''
        # test 데이터의 input/target 시퀀스 로드 및 정제
        test_data = pd.read_csv(parent_dir + '/data/' + my_dataset + '/test.csv', index_col=0)
        test_input_x = test_data['text']
        # test_input_x, trunc_len, target_idx = truncate_data(test_input_x, quantile=my_quantile)
        test_input_x_dict = my_tokenizer(list(test_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        test_input_x_np = test_input_x_dict['input_ids']
        test_input_x_att_mask_np = test_input_x_dict['attention_mask']

        test_target_y = test_data['label']
        # test_target_y = test_target_y.iloc[target_idx]
        test_target_y_np = np.array(test_target_y)

        '''
        데이터 저장
        '''
        # 저장경로 지정 및 생성
        save_dir = parent_dir + '/prep_data' + '/' + my_dataset + '/' + my_model
        createFolder(save_dir)

        # train/test input_x 시퀀스 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_input_x.npy'
        test_np_file_path = save_dir + '/test_input_x.npy'
        np.save(train_np_file_path, train_input_x_np)
        np.save(test_np_file_path, test_input_x_np)

        # train/test input_x 어텐션 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_input_att.npy'
        test_np_file_path = save_dir + '/test_input_att.npy'
        np.save(train_np_file_path, train_input_x_att_mask_np)
        np.save(test_np_file_path, test_input_x_att_mask_np)

        # train/test target_y 라벨 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_target_y.npy'
        test_np_file_path = save_dir + '/test_target_y.npy'
        np.save(train_np_file_path, train_target_y_np)
        np.save(test_np_file_path, test_target_y_np)

    else:
        '''
        토크나이저
        '''
        tokenizer_path = parent_dir + '/pretrained_weights/' + my_model
        left_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path + '/tokenizer_left')
        right_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path + '/tokenizer_right')

        '''
        train 데이터
        '''
        # train 데이터의 input 시퀀스 로드 및 정제
        train_data = pd.read_csv(parent_dir + '/data/' + my_dataset + '/train.csv', index_col=0)
        train_input_x = train_data['text']
        
        train_input_x_dict = left_tokenizer(list(train_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        train_input_x_np = train_input_x_dict['input_ids']

        if my_model == 'gpt2_small' or my_model == 'gpt2_large' or my_model == 'gpt_j' or my_model == 'opt' or my_model == 'xglm' or my_model == 'ctrl':
            train_input_x_att_mask_np = train_input_x_dict['attention_mask']

        # train 데이터의 y 시퀀스 로드 및 정제
        train_y = train_data['text']
        train_y = train_y.apply(lambda y : '<bos>' + y + '<eos>')
        # train_y = train_y.iloc[target_idx]
        train_y_dict = right_tokenizer(list(train_y), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        train_y_np = train_y_dict['input_ids']

        if my_model == 'gpt2_small' or my_model == 'gpt2_large' or my_model == 'gpt_j' or my_model == 'opt' or my_model == 'xglm' or my_model == 'ctrl':
            train_y_att_mask_np = train_y_dict['attention_mask']


        '''
        test 데이터
        '''
        # test 데이터의 input 시퀀스 로드 및 정제
        test_data = pd.read_csv(parent_dir + '/data/' + my_dataset + '/test.csv', index_col=0)
        test_input_x = test_data['text']
        # test_input_x, trunc_len, target_idx = truncate_data(test_input_x, quantile=my_quantile)
        test_input_x_dict = left_tokenizer(list(test_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        test_input_x_np = test_input_x_dict['input_ids']
    
        if my_model == 'gpt2_small' or my_model == 'gpt2_large' or my_model == 'gpt_j' or my_model == 'opt' or my_model == 'xglm' or my_model == 'ctrl':
            test_input_x_att_mask_np = test_input_x_dict['attention_mask']

        # test 데이터의 y 시퀀스 로드 및 정제
        test_y = test_data['text']
        test_y = test_y.apply(lambda y : '<bos>' + y + '<eos>')
        # test_y = test_y.iloc[target_idx]
        test_y_dict = right_tokenizer(list(test_y), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        test_y_np = test_y_dict['input_ids']
    
        if my_model == 'gpt2_small' or my_model == 'gpt2_large' or my_model == 'gpt_j' or my_model == 'opt' or my_model == 'xglm' or my_model == 'ctrl':
            test_y_att_mask_np = test_y_dict['attention_mask']


        '''
        데이터 저장
        '''
        # 저장경로 지정 및 생성
        save_dir = parent_dir + '/prep_data' + '/' + my_dataset + '/' + my_model
        createFolder(save_dir)

        # train/test input_x 시퀀스 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_input_x.npy'
        test_np_file_path = save_dir + '/test_input_x.npy'
        np.save(train_np_file_path, train_input_x_np)
        np.save(test_np_file_path, test_input_x_np)

        if my_model == 'gpt2_small' or my_model == 'gpt2_large' or my_model == 'gpt_j' or my_model == 'opt' or my_model == 'xglm' or my_model == 'ctrl':
            # train/test input_x 어텐션 데이터 .np 파일로 저장
            train_np_file_path = save_dir + '/train_input_att.npy'
            test_np_file_path = save_dir + '/test_input_att.npy'
            np.save(train_np_file_path, train_input_x_att_mask_np)
            np.save(test_np_file_path, test_input_x_att_mask_np)

        # train/test target_y 시퀀스 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_y.npy'
        test_np_file_path = save_dir + '/test_y.npy'
        np.save(train_np_file_path, train_y_np)
        np.save(test_np_file_path, test_y_np)

        if my_model == 'gpt2_small' or my_model == 'gpt2_large' or my_model == 'gpt_j' or my_model == 'opt' or my_model == 'xglm' or my_model == 'ctrl':
            # train/test target_y 어텐션 데이터 .np 파일로 저장
            train_np_file_path = save_dir + '/train_y_att.npy'
            test_np_file_path = save_dir + '/test_y_att.npy'
            np.save(train_np_file_path, train_y_att_mask_np)
            np.save(test_np_file_path, test_y_att_mask_np)

elif 'toxicity' in my_dataset:
    '''
    유해성 시퀀스 - 라벨 데이터 구축
    - 유해성 분류기를 훈련시키기 위한 데이터
    - 유해성 : non-toxic (0) / toxic (1) 
    '''

    if my_model == 'bert':

        '''
        토크나이저
        '''
        tokenizer_path = parent_dir + '/pretrained_weights/' + my_model
        my_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path + '/tokenizer', truncation = True, padding = 'max_length')

        '''
        train 데이터
        '''
        # train 데이터의 input/target 시퀀스 로드 및 정제
        train_data = pd.read_csv(parent_dir + '/data/' + my_dataset + '/train.csv', index_col=0)
        train_input_x = train_data['comment_text']
        
        train_input_x_dict = my_tokenizer(list(train_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        train_input_x_np = train_input_x_dict['input_ids']
        train_input_x_att_mask_np = train_input_x_dict['attention_mask']

        train_target_y = train_data['toxic']
        # train_target_y = train_target_y.iloc[target_idx]
        train_target_y_np = np.array(train_target_y)

        '''
        test 데이터
        '''
        # test 데이터의 input/target 시퀀스 로드 및 정제
        test_data = pd.read_csv(parent_dir + '/data/' + my_dataset + '/test.csv', index_col=0)
        test_data = test_data[test_data['toxic'] != -1]  # -1이라는 이상한 값이 들어가 있으므로 빼주어야 함.
        test_input_x = test_data['comment_text']

        # test_input_x, trunc_len, target_idx = truncate_data(test_input_x, quantile=my_quantile)
        test_input_x_dict = my_tokenizer(list(test_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        test_input_x_np = test_input_x_dict['input_ids']
        test_input_x_att_mask_np = test_input_x_dict['attention_mask']

        test_target_y = test_data['toxic']
        # test_target_y = test_target_y.iloc[target_idx]
        test_target_y_np = np.array(test_target_y)

        '''
        데이터 저장
        '''
        # 저장경로 지정 및 생성
        save_dir = parent_dir + '/prep_data' + '/' + my_dataset + '/' + my_model
        createFolder(save_dir)

        # train/test input_x 시퀀스 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_input_x.npy'
        test_np_file_path = save_dir + '/test_input_x.npy'
        np.save(train_np_file_path, train_input_x_np)
        np.save(test_np_file_path, test_input_x_np)

        # train/test input_x 어텐션 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_input_att.npy'
        test_np_file_path = save_dir + '/test_input_att.npy'
        np.save(train_np_file_path, train_input_x_att_mask_np)
        np.save(test_np_file_path, test_input_x_att_mask_np)

        # train/test target_y 라벨 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_target_y.npy'
        test_np_file_path = save_dir + '/test_target_y.npy'
        np.save(train_np_file_path, train_target_y_np)
        np.save(test_np_file_path, test_target_y_np)

    else:
        '''
        토크나이저
        '''
        tokenizer_path = parent_dir + '/pretrained_weights/' + my_model
        left_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path + '/tokenizer_left')
        right_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path + '/tokenizer_right')

        '''
        train 데이터
        '''
        # train 데이터의 input 시퀀스 로드 및 정제
        train_data = pd.read_csv(parent_dir + '/data/' + my_dataset + '/train.csv', index_col=0)
        train_input_x = train_data['comment_text']
        
        train_input_x_dict = left_tokenizer(list(train_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        train_input_x_np = train_input_x_dict['input_ids']
        train_input_x_att_mask_np = train_input_x_dict['attention_mask']

        # train 데이터의 y 시퀀스 로드 및 정제
        train_y = train_data['comment_text']
        train_y = train_y.apply(lambda y : '<bos>' + y + '<eos>')
        # train_y = train_y.iloc[target_idx]
        train_y_dict = right_tokenizer(list(train_y), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        train_y_np = train_y_dict['input_ids']
        train_y_att_mask_np = train_y_dict['attention_mask']


        '''
        test 데이터
        '''
        # test 데이터의 input 시퀀스 로드 및 정제
        test_data = pd.read_csv(parent_dir + '/data/' + my_dataset + '/test.csv', index_col=0)
        test_data = test_data[test_data['toxic'] != -1]  # -1이라는 이상한 값이 들어가 있으므로 빼주어야 함.
        test_input_x = test_data['comment_text']

        # test_input_x, trunc_len, target_idx = truncate_data(test_input_x, quantile=my_quantile)
        test_input_x_dict = left_tokenizer(list(test_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        test_input_x_np = test_input_x_dict['input_ids']
        test_input_x_att_mask_np = test_input_x_dict['attention_mask']

        # test 데이터의 y 시퀀스 로드 및 정제
        test_y = test_data['comment_text']
        test_y = test_y.apply(lambda y : '<bos>' + y + '<eos>')
        # test_y = test_y.iloc[target_idx]
        test_y_dict = right_tokenizer(list(test_y), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        test_y_np = test_y_dict['input_ids']
        test_y_att_mask_np = test_y_dict['attention_mask']


        '''
        데이터 저장
        '''
        # 저장경로 지정 및 생성
        save_dir = parent_dir + '/prep_data' + '/' + my_dataset + '/' + my_model
        createFolder(save_dir)

        # train/test input_x 시퀀스 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_input_x.npy'
        test_np_file_path = save_dir + '/test_input_x.npy'
        np.save(train_np_file_path, train_input_x_np)
        np.save(test_np_file_path, test_input_x_np)

        # train/test input_x 어텐션 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_input_att.npy'
        test_np_file_path = save_dir + '/test_input_att.npy'
        np.save(train_np_file_path, train_input_x_att_mask_np)
        np.save(test_np_file_path, test_input_x_att_mask_np)

        # train/test target_y 시퀀스 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_y.npy'
        test_np_file_path = save_dir + '/test_y.npy'
        np.save(train_np_file_path, train_y_np)
        np.save(test_np_file_path, test_y_np)

        # train/test target_y 어텐션 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_y_att.npy'
        test_np_file_path = save_dir + '/test_y_att.npy'
        np.save(train_np_file_path, train_y_att_mask_np)
        np.save(test_np_file_path, test_y_att_mask_np)


elif 'emotion' in my_dataset:
    '''
    감정 시퀀스 - 라벨 데이터 구축
    - 감정 분류기를 훈련시키기 위한 데이터
    - 감정 : no-emotion (0) / anger (1) / disgust (2) / fear (3) / happiness (4) / sadness (5) / surprise (6)

    참고)
    'emotion'과 'act'는 'dialog' 데이터셋에 같이 포함되어 있음. 따라서 my_dataset 데이터 로드 주소를 'dialog' 폴더로 지정해야함.
    'query' (또는 'response')을 시퀀스로 하고 그에 대응되는 emo_q (또는 emo_r)를 라벨로 사용함.
    '''
    N_turn = None

    if my_model == 'bert':
        '''
        토크나이저
        '''
        tokenizer_path = parent_dir + '/pretrained_weights/' + my_model
        my_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path + '/tokenizer', truncation = True, padding = 'max_length')

        '''
        train 데이터
        '''
        # train 데이터의 input/target 시퀀스 로드 및 정제
        train_data = pd.read_csv(parent_dir + '/data/' + 'dialog' + '/train_{}.csv'.format(N_turn), index_col=0)
        train_input_x = train_data['query']
        
        train_input_x_dict = my_tokenizer(list(train_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        train_input_x_np = train_input_x_dict['input_ids']
        train_input_x_att_mask_np = train_input_x_dict['attention_mask']

        train_target_y = train_data['emo_q']
        # train_target_y = train_target_y.iloc[target_idx]
        train_target_y_np = np.array(train_target_y)

        '''
        test 데이터
        '''
        # test 데이터의 input/target 시퀀스 로드 및 정제
        test_data = pd.read_csv(parent_dir + '/data/' + 'dialog' + '/test_{}.csv'.format(N_turn), index_col=0)
        test_input_x = test_data['query']
        # test_input_x, trunc_len, target_idx = truncate_data(test_input_x, quantile=my_quantile)
        test_input_x_dict = my_tokenizer(list(test_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        test_input_x_np = test_input_x_dict['input_ids']
        test_input_x_att_mask_np = test_input_x_dict['attention_mask']

        test_target_y = test_data['emo_q']
        # test_target_y = test_target_y.iloc[target_idx]
        test_target_y_np = np.array(test_target_y)

        '''
        데이터 저장
        '''
        # 저장경로 지정 및 생성
        save_dir = parent_dir + '/prep_data' + '/' + my_dataset + '/' + my_model
        createFolder(save_dir)

        # train/test input_x 시퀀스 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_input_x.npy'
        test_np_file_path = save_dir + '/test_input_x.npy'
        np.save(train_np_file_path, train_input_x_np)
        np.save(test_np_file_path, test_input_x_np)

        # train/test input_x 어텐션 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_input_att.npy'
        test_np_file_path = save_dir + '/test_input_att.npy'
        np.save(train_np_file_path, train_input_x_att_mask_np)
        np.save(test_np_file_path, test_input_x_att_mask_np)

        # train/test target_y 라벨 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_target_y.npy'
        test_np_file_path = save_dir + '/test_target_y.npy'
        np.save(train_np_file_path, train_target_y_np)
        np.save(test_np_file_path, test_target_y_np)

    else:
        '''
        토크나이저
        '''
        tokenizer_path = parent_dir + '/pretrained_weights/' + my_model
        left_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path + '/tokenizer_left')
        right_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path + '/tokenizer_right')

        '''
        train 데이터
        '''
        # train 데이터의 input 시퀀스 로드 및 정제
        train_data = pd.read_csv(parent_dir + '/data/' + 'dialog' + '/train_{}.csv'.format(N_turn), index_col=0)
        train_input_x = train_data['query']
        
        train_input_x_dict = left_tokenizer(list(train_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        train_input_x_np = train_input_x_dict['input_ids']
        train_input_x_att_mask_np = train_input_x_dict['attention_mask']

        # train 데이터의 y 시퀀스 로드 및 정제
        train_y = train_data['query']
        train_y = train_y.apply(lambda y : '<bos>' + y + '<eos>')
        # train_y = train_y.iloc[target_idx]
        train_y_dict = right_tokenizer(list(train_y), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        train_y_np = train_y_dict['input_ids']
        train_y_att_mask_np = train_y_dict['attention_mask']


        '''
        test 데이터
        '''
        # test 데이터의 input 시퀀스 로드 및 정제
        test_data = pd.read_csv(parent_dir + '/data/' + 'dialog' + '/test_{}.csv'.format(N_turn), index_col=0)
        test_input_x = test_data['query']
        # test_input_x, trunc_len, target_idx = truncate_data(test_input_x, quantile=my_quantile)
        test_input_x_dict = left_tokenizer(list(test_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        test_input_x_np = test_input_x_dict['input_ids']
        test_input_x_att_mask_np = test_input_x_dict['attention_mask']

        # test 데이터의 y 시퀀스 로드 및 정제
        test_y = test_data['query']
        test_y = test_y.apply(lambda y : '<bos>' + y + '<eos>')
        # test_y = test_y.iloc[target_idx]
        test_y_dict = right_tokenizer(list(test_y), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        test_y_np = test_y_dict['input_ids']
        test_y_att_mask_np = test_y_dict['attention_mask']


        '''
        데이터 저장
        '''
        # 저장경로 지정 및 생성
        save_dir = parent_dir + '/prep_data' + '/' + my_dataset + '/' + my_model
        createFolder(save_dir)

        # train/test input_x 시퀀스 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_input_x.npy'
        test_np_file_path = save_dir + '/test_input_x.npy'
        np.save(train_np_file_path, train_input_x_np)
        np.save(test_np_file_path, test_input_x_np)

        # train/test input_x 어텐션 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_input_att.npy'
        test_np_file_path = save_dir + '/test_input_att.npy'
        np.save(train_np_file_path, train_input_x_att_mask_np)
        np.save(test_np_file_path, test_input_x_att_mask_np)

        # train/test target_y 시퀀스 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_y.npy'
        test_np_file_path = save_dir + '/test_y.npy'
        np.save(train_np_file_path, train_y_np)
        np.save(test_np_file_path, test_y_np)

        # train/test target_y 어텐션 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_y_att.npy'
        test_np_file_path = save_dir + '/test_y_att.npy'
        np.save(train_np_file_path, train_y_att_mask_np)
        np.save(test_np_file_path, test_y_att_mask_np)


elif 'act' in my_dataset:
    '''
    대화행위 시퀀스 - 라벨 데이터 구축
    - 대화행위 분류기를 훈련시키기 위한 데이터
    - 대화행위 : __dummy__ (0) / inform (1) / question (2) / directive (3) / commissive (4)

    참고)
    'emotion'과 'act'는 'dialog' 데이터셋에 같이 포함되어 있음. 따라서 my_dataset 데이터 로드 주소를 'dialog' 폴더로 지정해야함.
    'query' (또는 'response')을 시퀀스로 하고 그에 대응되는 emo_q (또는 emo_r)를 라벨로 사용함.
    '''
    N_turn = None

    if my_model == 'bert':
        '''
        토크나이저
        '''
        tokenizer_path = parent_dir + '/pretrained_weights/' + my_model
        my_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path + '/tokenizer', truncation = True, padding = 'max_length')

        '''
        train 데이터
        '''
        # train 데이터의 input/target 시퀀스 로드 및 정제
        train_data = pd.read_csv(parent_dir + '/data/' + 'dialog' + '/train_{}.csv'.format(N_turn), index_col=0)
        train_input_x = train_data['query']
        
        train_input_x_dict = my_tokenizer(list(train_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        train_input_x_np = train_input_x_dict['input_ids']
        train_input_x_att_mask_np = train_input_x_dict['attention_mask']

        train_target_y = train_data['act_q']
        # train_target_y = train_target_y.iloc[target_idx]
        train_target_y_np = np.array(train_target_y)


        '''
        test 데이터
        '''
        # test 데이터의 input/target 시퀀스 로드 및 정제
        test_data = pd.read_csv(parent_dir + '/data/' + 'dialog' + '/test_{}.csv'.format(N_turn), index_col=0)
        test_input_x = test_data['query']
        # test_input_x, trunc_len, target_idx = truncate_data(test_input_x, quantile=my_quantile)
        test_input_x_dict = my_tokenizer(list(test_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        test_input_x_np = test_input_x_dict['input_ids']
        test_input_x_att_mask_np = test_input_x_dict['attention_mask']

        test_target_y = test_data['act_q']
        # test_target_y = test_target_y.iloc[target_idx]
        test_target_y_np = np.array(test_target_y)

        '''
        데이터 생성
        '''
        # 저장경로 지정 및 생성
        save_dir = parent_dir + '/prep_data' + '/' + my_dataset + '/' + my_model
        createFolder(save_dir)

        # train/test input_x 시퀀스 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_input_x.npy'
        test_np_file_path = save_dir + '/test_input_x.npy'
        np.save(train_np_file_path, train_input_x_np)
        np.save(test_np_file_path, test_input_x_np)

        # train/test input_x 어텐션 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_input_att.npy'
        test_np_file_path = save_dir + '/test_input_att.npy'
        np.save(train_np_file_path, train_input_x_att_mask_np)
        np.save(test_np_file_path, test_input_x_att_mask_np)

        # train/test target_y 라벨 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_target_y.npy'
        test_np_file_path = save_dir + '/test_target_y.npy'
        np.save(train_np_file_path, train_target_y_np)
        np.save(test_np_file_path, test_target_y_np)

    else:
        '''
        토크나이저
        '''
        tokenizer_path = parent_dir + '/pretrained_weights/' + my_model
        left_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path + '/tokenizer_left')
        right_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path + '/tokenizer_right')

        '''
        train 데이터
        '''
        # train 데이터의 input 시퀀스 로드 및 정제
        train_data = pd.read_csv(parent_dir + '/data/' + 'dialog' + '/train_{}.csv'.format(N_turn), index_col=0)
        train_input_x = train_data['query']
        
        train_input_x_dict = left_tokenizer(list(train_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        train_input_x_np = train_input_x_dict['input_ids']
        train_input_x_att_mask_np = train_input_x_dict['attention_mask']

        # train 데이터의 y 시퀀스 로드 및 정제
        train_y = train_data['query']
        train_y = train_y.apply(lambda y : '<bos>' + y + '<eos>')
        # train_y = train_y.iloc[target_idx]
        train_y_dict = right_tokenizer(list(train_y), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        train_y_np = train_y_dict['input_ids']
        train_y_att_mask_np = train_y_dict['attention_mask']


        '''
        test 데이터
        '''
        # test 데이터의 input 시퀀스 로드 및 정제
        test_data = pd.read_csv(parent_dir + '/data/' + 'dialog' + '/test_{}.csv'.format(N_turn), index_col=0)
        test_input_x = test_data['query']
        # test_input_x, trunc_len, target_idx = truncate_data(test_input_x, quantile=my_quantile)
        test_input_x_dict = left_tokenizer(list(test_input_x), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        test_input_x_np = test_input_x_dict['input_ids']
        test_input_x_att_mask_np = test_input_x_dict['attention_mask']

        # test 데이터의 y 시퀀스 로드 및 정제
        test_y = test_data['query']
        test_y = test_y.apply(lambda y : '<bos>' + y + '<eos>')
        # test_y = test_y.iloc[target_idx]
        test_y_dict = right_tokenizer(list(test_y), return_tensors='np', truncation=True, max_length=my_max_len, padding=True)         # 인풋 시퀀스 토크나이징
        test_y_np = test_y_dict['input_ids']
        test_y_att_mask_np = test_y_dict['attention_mask']


        '''
        데이터 저장
        '''
        # 저장경로 지정 및 생성
        save_dir = parent_dir + '/prep_data' + '/' + my_dataset + '/' + my_model
        createFolder(save_dir)

        # train/test input_x 시퀀스 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_input_x.npy'
        test_np_file_path = save_dir + '/test_input_x.npy'
        np.save(train_np_file_path, train_input_x_np)
        np.save(test_np_file_path, test_input_x_np)

        # train/test input_x 어텐션 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_input_att.npy'
        test_np_file_path = save_dir + '/test_input_att.npy'
        np.save(train_np_file_path, train_input_x_att_mask_np)
        np.save(test_np_file_path, test_input_x_att_mask_np)

        # train/test target_y 시퀀스 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_y.npy'
        test_np_file_path = save_dir + '/test_y.npy'
        np.save(train_np_file_path, train_y_np)
        np.save(test_np_file_path, test_y_np)

        # train/test target_y 어텐션 데이터 .np 파일로 저장
        train_np_file_path = save_dir + '/train_y_att.npy'
        test_np_file_path = save_dir + '/test_y_att.npy'
        np.save(train_np_file_path, train_y_att_mask_np)
        np.save(test_np_file_path, test_y_att_mask_np)
