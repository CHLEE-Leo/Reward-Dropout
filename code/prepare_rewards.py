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
from utils import createFolder
from tensorflow.keras.utils import Progbar
from transformers import AutoTokenizer, TFBertModel


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
GPU 셋업
'''
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
initialize_setting()
seed_everything(47)

'''
파라미터 로드
'''
parser = argparse.ArgumentParser(description='receive the parameters')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--ref_model_name', type=str, default='opt_large')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--decoding', type=str, default='beam')
args = parser.parse_args()

my_dataset = args.dataset
ref_model_name = args.ref_model_name
my_batch_size = args.batch_size
my_decoding = args.decoding
# my_dataset = 'sentiment-1'
# ref_model_name = 'gpt2_small'
# my_batch_size = 256
# my_decoding = 'stochastic'


'''
데이터 로드 경로 설정
'''
# 현 위치의 부모위치 확인
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
pretrained_config_dir = parent_dir + '/pretrained_weights/bert'
pretrained_tokenizer_dir = pretrained_config_dir + '/tokenizer'
pretrained_weights_dir = pretrained_config_dir + '/model'
bert_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_dir)
bert_model = TFBertModel.from_pretrained(pretrained_weights_dir)
bert_model.resize_token_embeddings(len(bert_tokenizer))
bert_model = BERT_Classifier(bert_model, num_labels)

# 파인튜닝 가중치 로드
bert_bs = 256
finetuned_weights_dir = parent_dir + '/weights' + '/' + my_dataset.split('-')[0] + '/bert'
my_model_ft_weights_dir = glob.glob(finetuned_weights_dir + '/*{}*'.format(bert_bs))[0]
bert_model.load_weights(tf.train.latest_checkpoint(my_model_ft_weights_dir))
bert_model.trainable = False

'''
모든 케이스의 사전생성 데이터 로드
'''
# gen_seq_{}.txt 파일들의 경로 리스트
gen_texts_dir = glob.glob(parent_dir + '/prep_data' + '/' + my_dataset + '/' + ref_model_name + '/' + '*gen_seq_{}*[0-9].txt'.format(my_decoding))


'''
생성결과 저장 경로 지정
'''
GEN_SAVE_DIR = parent_dir + '/prep_data' + '/' + my_dataset + '/' + ref_model_name
createFolder(GEN_SAVE_DIR)
print('GEN_SAVE_DIR :', GEN_SAVE_DIR)


'''
각 생성 케이스 마다 보상 계산하기
'''
full_target_rewards_list = []
for gen_case_num, each_case in enumerate(gen_texts_dir):

    # if gen_case_num > 0:
    #     break;

    # 각 line이 하나의 element인 list 형식으로 파일 열기
    with open(each_case, 'r') as file:
        gen_texts = file.readlines()
        
    # 버트 토크나이징 해주기
    gen_texts_bert_tokenized = bert_tokenizer(gen_texts, return_tensors='np', truncation=True, padding=True)     # <bos> 포함
    gen_texts_encoded = gen_texts_bert_tokenized['input_ids']
    gen_texts_masks = gen_texts_bert_tokenized['attention_mask']        

    # 텐서 변환 + 배치 슬라이싱
    with tf.device("/cpu:0"):

        # 텐서 변환
        # gen_dat = tf.data.Dataset.from_tensor_slices((gen_texts_encoded, gen_texts_masks)).shuffle(buffer_size = gen_texts_encoded.shape[0], reshuffle_each_iteration = False)
        gen_dat = tf.data.Dataset.from_tensor_slices((gen_texts_encoded, gen_texts_masks)) # gen_seq랑 reward의 pair를 맞춰주려면 절대로 셔플하면 안됨. 

        # 배치 슬라이싱
        gen_batch = gen_dat.batch(batch_size=my_batch_size, drop_remainder=True)

    # 생성 시간 및 진행상황 측정
    start_time = time.time()
    total_gen_case_num_num = len(gen_texts_dir)
    print("\n decoding : {} \n num_gen_case_num : {}/{}".format(my_decoding, gen_case_num + 1, total_gen_case_num_num))
    pb_i = Progbar(len(gen_batch))

    # 배치마다 보상 계산하기
    for idx, (seqs, masks) in enumerate(gen_batch):

        # 보상 계산
        rewards = bert_model(seqs, attention_mask=masks, training=False)
        target_style_indicator = int(my_dataset.split('-')[1])              # e.g., in case of sentiment dataset, the target style indicator consists of 0 and 1.
        target_rewards_list = list(tf.cast(rewards, dtype=tf.float16)[:, target_style_indicator].numpy())

        # 배치 통합하기
        if idx == 0:
            total_target_rewards_list = copy.deepcopy(target_rewards_list)
        else:
            total_target_rewards_list += target_rewards_list

        # 진행상태 바 (Progress Bar) 업데이트
        pb_i.update(idx+1)

    # .txt.포맷으로 저장하기 위해 "list of floats -> list of string" 변환하기
    converted_total_target_rewards_list = ['{:.3f}'.format(x) for x in total_target_rewards_list]

    # 생성 케이스 별 예측된 보상들 텍스트 파일로 저장하기
    reward_txt_file_path = GEN_SAVE_DIR + '/train_reward_{}_{}.txt'.format(my_decoding, gen_case_num)
    with open(reward_txt_file_path, 'a') as file:
        file.write('\n'.join(converted_total_target_rewards_list))

    # 전체 생성 케이스 통합하기
    full_target_rewards_list += total_target_rewards_list

# .txt.포맷으로 저장하기 위해 "list of floats -> list of string" 변환하기
converted_full_target_rewards_list = ['{:.3f}'.format(x) for x in full_target_rewards_list]

# 전체 케이스에 대해서 예측된 보상들 텍스트 파일로 저장하기
reward_txt_file_path = GEN_SAVE_DIR + '/train_reward_{}_full.txt'.format(my_decoding)
with open(reward_txt_file_path, 'a') as file:
    file.write('\n'.join(converted_full_target_rewards_list))
