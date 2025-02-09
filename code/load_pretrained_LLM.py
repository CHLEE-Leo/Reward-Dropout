import os
from pathlib import Path
import argparse
import copy
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from transformers import AutoTokenizer, TFAutoModelForCausalLM, TFBertModel, TFTransfoXLLMHeadModel, TFGPTJForCausalLM, TFT5ForConditionalGeneration
import glob

from utils import get_params

# from pretrained_model import BERT_Classifier

parent_dir = str(Path(os.getcwd()).parents[0])

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
        # label_outputs = outputs.last_hidden_state[:, 0, :]      # (batch_size, n_dim)
        label_outputs = outputs.pooler_output                       # (batch_size, n_dim)
        label_outputs = self.dropout(label_outputs, training = True)
        label_preds = self.classifier(label_outputs)            # (batch_size, num_attris)

        return label_preds


'''
파라미터 로드
'''
args = get_params()
my_task = args.task


'''
모델 및 토크나이저 로드
'''
# --task == 'ft' 일 경우 
if my_task == 'ft':

    my_dataset = args.dataset
    my_model = args.model
    my_lr = args.lr
    my_bs = args.batch_size
    my_epoch = args.num_epoch

    # --model == gpt 계열인 경우
    if my_model in ['gpt2_small', 'gpt2_large', 'dialoGPT']:

        '''
        pretrained LLM 모델 로드하기
        '''
        if my_model == 'gpt2_small':

            '''
            저장 및 호출경로 생성
            '''
            GPT_SMALL_SAVE_PATH = parent_dir + '/pretrained_weights/gpt2_small'

            '''
            GPT2 모델 및 토크나이저 로드
            - tokenizer : gpt2 토크나이저
            - model : Huggingface에서 제공하는 텐서플로 버젼 gpt2-small 사전훈련 모델
            - resize_token_embeddings() : 허깅페이스 사전훈련 모델을 내려받을 때 (set_pretrained_LLM.py), <bos>, <eos> 등 special_token들을 추가해준 바 있음. 이것을 반영하여 token_embedding을 resize. 
            '''
            tokenizer_right = AutoTokenizer.from_pretrained(GPT_SMALL_SAVE_PATH + '/tokenizer_right')
            tokenizer_right = AutoTokenizer.from_pretrained(GPT_SMALL_SAVE_PATH + '/tokenizer_right')
            model = TFAutoModelForCausalLM.from_pretrained(GPT_SMALL_SAVE_PATH + '/model')
            model.resize_token_embeddings(len(tokenizer_right))

        elif my_model == 'gpt2_large':

            '''
            저장 및 호출경로 생성
            '''
            GPT_LARGE_SAVE_PATH = parent_dir + '/pretrained_weights/gpt2_large'

            '''
            GPT2 모델 및 토크나이저 로드
            - tokenizer : gpt2-large 토크나이저
            - model : Huggingface에서 제공하는 텐서플로 버젼 gpt2-large 사전훈련 모델
            - resize_token_embeddings() : 허깅페이스 사전훈련 모델을 내려받을 때 (set_pretrained_LLM.py), <bos>, <eos> 등 special_token들을 추가해준 바 있음. 이것을 반영하여 token_embedding을 resize. 
            '''
            tokenizer_right = AutoTokenizer.from_pretrained(GPT_LARGE_SAVE_PATH + '/tokenizer_right')
            tokenizer_right = AutoTokenizer.from_pretrained(GPT_LARGE_SAVE_PATH + '/tokenizer_right')
            model = TFAutoModelForCausalLM.from_pretrained(GPT_LARGE_SAVE_PATH + '/model')
            model.resize_token_embeddings(len(tokenizer_right))

        elif my_model == 'dialoGPT':
            '''
            저장 및 호출경로 생성
            '''
            DialoGPT_SAVE_PATH = parent_dir + '/pretrained_weights/dialoGPT'

            '''
            GPT2 모델 및 토크나이저 로드
            - tokenizer : dialoGPT 토크나이저
            - model : Huggingface에서 제공하는 텐서플로 버젼 dialoGPT 사전훈련 모델
            - resize_token_embeddings() : 허깅페이스 사전훈련 모델을 내려받을 때 (set_pretrained_LLM.py), <bos>, <eos> 등 special_token들을 추가해준 바 있음. 이것을 반영하여 token_embedding을 resize. 
            '''
            tokenizer_right = AutoTokenizer.from_pretrained(DialoGPT_SAVE_PATH + '/tokenizer_right')
            tokenizer_right = AutoTokenizer.from_pretrained(DialoGPT_SAVE_PATH + '/tokenizer_right')
            model = TFAutoModelForCausalLM.from_pretrained(DialoGPT_SAVE_PATH + '/model')
            model.resize_token_embeddings(len(tokenizer_right))

    # --model =='bert' 일 경우
    elif my_model == 'bert':

        '''
        pretrained BERT 모델 로드하기
        '''
        if my_model == 'bert':

            '''
            데이터셋마다 라벨 갯수가 다름
            - emotion 데이터셋 : 라벨 7개
            --> (4) 감정 : no-emotion (0) / anger (1) / disgust (2) / fear (3) / happiness (4) / sadness (5) / surprise (6)

            - act 데이터셋 : 라벨 5 개
            --> (5) 대화행위 : __dommy__ (0) / inform (1) / question (2) / directive (3) / commissive (4)

            - 그 외 : 라벨 2 개
            --> (1) 긍부정 : negative (0) / postive (1)
            --> (2) 공손함 : polite (P9 = 1) / non-polite (P0-P8 = 0)
            --> (3) 유해성 : non-toxic (0) / toxic (1) 
            --> (6) 자연스러움 :

            '''
            if 'emotion' in my_dataset:
                num_labels = 7    

            elif 'act' in my_dataset:
                num_labels = 5

            elif 'topic' in my_dataset:
                num_labels = 4

            else:
                num_labels = 2


            '''
            저장 및 호출경로 생성
            - BERT_SAVE_PATH : Huggingface에서 제공하는 Bert 모델의 토크나이저와 가중치가 존재하는 경로
            '''
            BERT_SAVE_PATH = parent_dir + '/pretrained_weights/bert'

            '''
            BERT 모델 및 토크나이저 로드
            - tokenizer : 버트 토크나이저
            - model : Huggingface에서 제공하는 텐서플로 버젼 bert 사전훈련 모델
            - bert_classifier : 
            '''
            tokenizer = AutoTokenizer.from_pretrained(BERT_SAVE_PATH + '/tokenizer')
            model = TFBertModel.from_pretrained(BERT_SAVE_PATH + '/model')
            model.resize_token_embeddings(len(tokenizer))
            model = BERT_Classifier(model, num_labels)

# --task == 'rl' 일 경우
elif my_task == 'rl':

    my_dataset = args.dataset
    my_model = args.model
    # my_lr = args.lr
    # my_bs = args.batch_size
    # my_epoch = args.num_epoch    

    '''
    task == 'ft'를 통해 파인튜닝 된 GPT2와 BERT를 모두 불러오기
    - GPT2는 model == '' 인자 설정에 따라 다른 모델이 로드됨 (e.g., gpt2-small, gpt2-large, trans_xl).
    - BERT는 dataset =='' 인자 설정에 따라 다른 모델이 로드됨 (e.g., sentiment, toxicity, politeness).
    '''
    if my_model == 'gpt2_small_init_weight':
        my_init_weight = args.init_weight   # my_model == gpt2_small_init_weight 에서만 사용.

        '''
        행동 정책 모델 (behavior policy model)
        - 행동정책에는 .resize_token_embeddings()를 안하는게 확실히 맞음 !
        - 실험적으로 확인하였음. .resize_token_embeddings() 하면 gpt_model.generate()에서 이상하게 생성됨.
        '''
        # 토크나이저 로드 및 모델 초기화
        pretraind_config_dir = parent_dir + '/pretrained_weights' + '/' + my_model.replace('_init_weight', '')
        pretrained_tokenizer_dir = pretraind_config_dir + '/tokenizer_right'
        pretrained_weights_dir = pretraind_config_dir + '/model'
        gpt_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_dir)
        gpt_model = TFAutoModelForCausalLM.from_pretrained(pretrained_weights_dir)
        # gpt_model.resize_token_embeddings(len(gpt_tokenizer))

        '''
        타겟 정책 모델 (target policy model)
        - 타겟정책에는 .resize_token_embeddings()를 하는데 맞음 !
        '''
        # 토크나이저 로드 및 모델 초기화
        pretraind_config_dir = parent_dir + '/pretrained_weights' + '/' + my_model.replace('_init_weight', '')
        pretrained_tokenizer_dir = pretraind_config_dir + '/tokenizer_right'
        pretrained_weights_dir = pretraind_config_dir + '/model'
        gpt_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_dir)
        target_model = TFAutoModelForCausalLM.from_pretrained(pretrained_weights_dir)
        target_model.resize_token_embeddings(len(gpt_tokenizer))


        # 타겟 정책 (tp) 모델의 훈련 가능한 가중치를 초기화
        # - 이렇게 해야 gpt2_small이 not_pretrained_tp가 됨.
        # for my_init_weight in ['uniform', 'normal', 'Glorot', 'He']:
        for weight in target_model.trainable_weights:
            if hasattr(weight, 'shape'):

                if my_init_weight == 'uniform':
                    weight.assign(tf.random.uniform(weight.shape, minval=-0.05, maxval=0.05))

                elif my_init_weight == 'normal':
                    weight.assign(tf.random.normal(weight.shape, mean=0.0, stddev=0.05))

                elif my_init_weight == 'Glorot':
                    initializer = tf.initializers.GlorotNormal()
                    weight.assign(initializer(weight.shape))

                elif my_init_weight == 'He':
                    initializer = tf.initializers.HeNormal()
                    weight.assign(initializer(weight.shape))

                elif my_init_weight == 'constant':
                    weight.assign(tf.constant(0.01, shape=weight.shape))


    elif my_model == 'gpt2_small' or my_model == 'gpt2_large':
        # 1) GPT2의 토크나이저 로드 및 모델 초기화 & 미세조정 가중치 로드
        '''
        행동 정책 모델 (behavior policy model)
        - 행동정책에는 .resize_token_embeddings()를 안하는게 확실히 맞음 !
        '''
        # 토크나이저 로드 및 모델 초기화
        pretraind_config_dir = parent_dir + '/pretrained_weights' + '/' + my_model
        pretrained_tokenizer_dir = pretraind_config_dir + '/tokenizer_right'
        pretrained_weights_dir = pretraind_config_dir + '/model'
        gpt_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_dir)
        gpt_model = TFAutoModelForCausalLM.from_pretrained(pretrained_weights_dir, from_pt=True)
        # gpt_model.resize_token_embeddings(len(gpt_tokenizer))


        '''
        타겟 정책 모델 (target policy model)
        - 타겟정책에는 .resize_token_embeddings()를 하는데 맞음 !
        '''
        # 토크나이저 로드 및 모델 초기화
        pretraind_config_dir = parent_dir + '/pretrained_weights' + '/' + my_model
        pretrained_tokenizer_dir = pretraind_config_dir + '/tokenizer_right'
        pretrained_weights_dir = pretraind_config_dir + '/model'
        gpt_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_dir)
        target_model = TFAutoModelForCausalLM.from_pretrained(pretrained_weights_dir, from_pt=True)
        target_model.resize_token_embeddings(len(gpt_tokenizer))


    elif my_model == 'opt' or my_model == 'opt_large' or my_model == 'xglm' or my_model == 'xglm_large':
        '''
        행동 정책 모델 (behavior policy model)
        - 행동정책에는 .resize_token_embeddings()를 안하는게 확실히 맞음 !
        '''
        # 토크나이저 로드 및 모델 초기화
        pretraind_config_dir = parent_dir + '/pretrained_weights' + '/' + my_model
        pretrained_tokenizer_dir = pretraind_config_dir + '/tokenizer_right'
        pretrained_weights_dir = pretraind_config_dir + '/model'
        gpt_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_dir)
        gpt_model = TFAutoModelForCausalLM.from_pretrained(pretrained_weights_dir, from_pt=True)
        # gpt_model.resize_token_embeddings(len(gpt_tokenizer))


        '''
        타겟 정책 모델 (target policy model)
        - 타겟정책에는 .resize_token_embeddings()를 하는데 맞음 !
        '''
        # 토크나이저 로드 및 모델 초기화
        pretraind_config_dir = parent_dir + '/pretrained_weights' + '/' + my_model
        pretrained_tokenizer_dir = pretraind_config_dir + '/tokenizer_right'
        pretrained_weights_dir = pretraind_config_dir + '/model'
        gpt_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_dir)
        target_model = TFAutoModelForCausalLM.from_pretrained(pretrained_weights_dir, from_pt=True)
        target_model.resize_token_embeddings(len(gpt_tokenizer))

    elif my_model == 'gpt_j':
        '''
        행동 정책 모델 (behavior policy model)
        - 행동정책에는 .resize_token_embeddings()를 안하는게 확실히 맞음 !
        '''
        # 토크나이저 로드 및 모델 초기화
        pretraind_config_dir = parent_dir + '/pretrained_weights' + '/' + my_model
        pretrained_tokenizer_dir = pretraind_config_dir + '/tokenizer_right'
        pretrained_weights_dir = pretraind_config_dir + '/model'
        gpt_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_dir)
        gpt_model = TFGPTJForCausalLM.from_pretrained(pretrained_weights_dir, from_pt=True)
        # gpt_model.resize_token_embeddings(len(gpt_tokenizer))


        '''
        타겟 정책 모델 (target policy model)
        - 타겟정책에는 .resize_token_embeddings()를 하는데 맞음 !
        '''
        # 토크나이저 로드 및 모델 초기화
        pretraind_config_dir = parent_dir + '/pretrained_weights' + '/' + my_model
        pretrained_tokenizer_dir = pretraind_config_dir + '/tokenizer_right'
        pretrained_weights_dir = pretraind_config_dir + '/model'
        gpt_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_dir)
        target_model = TFGPTJForCausalLM.from_pretrained(pretrained_weights_dir, from_pt=True)
        target_model.resize_token_embeddings(len(gpt_tokenizer))        

    # elif my_model == 'trans_xl':
    #     '''
    #     행동 정책 모델 (behavior policy model)
    #     - 행동정책에는 .resize_token_embeddings()를 안하는게 확실히 맞음 !
    #     '''
    #     # 토크나이저 로드 및 모델 초기화
    #     pretraind_config_dir = parent_dir + '/pretrained_weights' + '/' + my_model
    #     pretrained_tokenizer_dir = pretraind_config_dir + '/tokenizer_right'
    #     pretrained_weights_dir = pretraind_config_dir + '/model'
    #     gpt_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_dir)
    #     gpt_model = TFTransfoXLLMHeadModel.from_pretrained(pretrained_weights_dir, torch_dtype='auto')
    #     # gpt_model.resize_token_embeddings(len(gpt_tokenizer))

    #     '''
    #     타겟 정책 모델 (target policy model)
    #     - 타겟정책에는 .resize_token_embeddings()를 하는데 맞음 !
    #     '''
    #     # 토크나이저 로드 및 모델 초기화
    #     pretraind_config_dir = parent_dir + '/pretrained_weights' + '/' + my_model
    #     pretrained_tokenizer_dir = pretraind_config_dir + '/tokenizer_right'
    #     pretrained_weights_dir = pretraind_config_dir + '/model'
    #     gpt_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_dir)
    #     target_model = TFTransfoXLLMHeadModel.from_pretrained(pretrained_weights_dir, torch_dtype='auto')
    #     target_model.resize_token_embeddings(len(gpt_tokenizer))

    elif my_model == 't5' or my_model == 'flan_t5' or my_model == 'flan_ul2':
        '''
        행동 정책 모델 (behavior policy model)
        - 행동정책에는 .resize_token_embeddings()를 안하는게 확실히 맞음 !
        '''
        # 토크나이저 로드 및 모델 초기화
        pretraind_config_dir = parent_dir + '/pretrained_weights' + '/' + my_model
        pretrained_tokenizer_dir = pretraind_config_dir + '/tokenizer_right'
        pretrained_weights_dir = pretraind_config_dir + '/model'
        gpt_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_dir)
        gpt_model = TFT5ForConditionalGeneration.from_pretrained(pretrained_weights_dir,
                                                                 from_pt=True)
        # gpt_model.resize_token_embeddings(len(gpt_tokenizer))

        '''
        타겟 정책 모델 (target policy model)
        - 타겟정책에는 .resize_token_embeddings()를 하는데 맞음 !
        '''
        # 토크나이저 로드 및 모델 초기화
        pretraind_config_dir = parent_dir + '/pretrained_weights' + '/' + my_model
        pretrained_tokenizer_dir = pretraind_config_dir + '/tokenizer_right'
        pretrained_weights_dir = pretraind_config_dir + '/model'
        gpt_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_dir)
        target_model = TFT5ForConditionalGeneration.from_pretrained(pretrained_weights_dir,
                                                                    from_pt=True)
        target_model.resize_token_embeddings(len(gpt_tokenizer))        

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
    pretraind_config_dir = parent_dir + '/pretrained_weights/bert'
    pretrained_tokenizer_dir = pretraind_config_dir + '/tokenizer'
    pretrained_weights_dir = pretraind_config_dir + '/model'
    bert_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_dir)
    bert_model = TFBertModel.from_pretrained(pretrained_weights_dir)
    bert_model.resize_token_embeddings(len(bert_tokenizer))
    bert_model = BERT_Classifier(bert_model, num_labels)

    # 파인튜닝 가중치 로드
    bert_bs = 256
    finetuned_weights_dir = parent_dir + '/weights' + '/' + my_dataset.split('-')[0] + '/bert'
    my_model_ft_weights_dir = glob.glob(finetuned_weights_dir + '/*{}*'.format(bert_bs))[0]
    bert_model.load_weights(tf.train.latest_checkpoint(my_model_ft_weights_dir))

elif my_task == 'train_eval' or my_task == 'test_eval':

    my_dataset = args.dataset
    my_model = args.model
    my_decoding = args.decoding
    my_dropout = args.dropout
    my_dropout_rate = args.dropout_rate

    '''
    task == 'eval'를 통해 강화학습 된 GPT2와 BERT를 모두 불러오기
    - GPT2는 model == '' 인자 설정에 따라 다른 모델이 로드됨 (e.g., gpt2-small, gpt2-large, trans_xl).
    - BERT는 dataset =='' 인자 설정에 따라 다른 모델이 로드됨 (e.g., sentiment, toxicity, politeness).
    '''

    # 1) GPT2의 토크나이저 로드 및 모델 초기화 & 미세조정 가중치 로드
    '''
    타겟 정책 모델 (target policy model)
    '''
    # 평가 대상이 train_reward 인 경우
    if my_task == 'train_eval':

        # 토크나이저 로드 및 모델 초기화
        pretraind_config_dir = parent_dir + '/pretrained_weights' + '/' + my_model
        pretrained_tokenizer_dir = pretraind_config_dir + '/tokenizer_right'
        pretrained_weights_dir = pretraind_config_dir + '/model'
        gpt_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_dir)
        target_model = TFAutoModelForCausalLM.from_pretrained(pretrained_weights_dir)
        # target_model.resize_token_embeddings(len(gpt_tokenizer))

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
        pretraind_config_dir = parent_dir + '/pretrained_weights' + '/bert'
        pretrained_tokenizer_dir = pretraind_config_dir + '/tokenizer'
        pretrained_weights_dir = pretraind_config_dir + '/model'
        bert_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_dir)
        bert_model = TFBertModel.from_pretrained(pretrained_weights_dir)
        bert_model.resize_token_embeddings(len(bert_tokenizer))
        bert_model = BERT_Classifier(bert_model, num_labels)

        # 파인튜닝 가중치 로드
        finetuned_weights_dir = parent_dir + '/weights' + '/' + my_dataset.split('-')[0] + '/bert'
        my_model_ft_weights_dir = glob.glob(finetuned_weights_dir + '/*256*')[0]
        bert_model.load_weights(tf.train.latest_checkpoint(my_model_ft_weights_dir))

    # 평가 대상이 test_performance 인 경우
    elif my_task == 'test_eval':

        # 토크나이저 로드 및 모델 초기화
        pretraind_config_dir = parent_dir + '/pretrained_weights' + '/' + my_model
        pretrained_tokenizer_dir = pretraind_config_dir + '/tokenizer_right'
        pretrained_weights_dir = pretraind_config_dir + '/model'
        gpt_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_dir)
        target_model = TFAutoModelForCausalLM.from_pretrained(pretrained_weights_dir)

        # 훈련 가중치 주소 정의
        reinforced_weights_dir = parent_dir + '/weights' + '/' + my_dataset + '/' + my_model
        my_model_rl_weights_dir = glob.glob(reinforced_weights_dir + '/*{}**{}**{}'.format(my_decoding, my_dropout, my_dropout_rate))[0]
        print('my_model_rl_weights_dir :', my_model_rl_weights_dir)

        # 훈련 가중치 로드
        target_model.load_weights(tf.train.latest_checkpoint(my_model_rl_weights_dir))