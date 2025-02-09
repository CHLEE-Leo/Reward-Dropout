# %%
'''
test_eval 코드 ~
'''
import os
from pathlib import Path
import argparse
import copy
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import get_params
import random
import time

from transformers import AutoTokenizer, TFAutoModelForCausalLM, TFBertModel
import glob

# # 필수 인자들 설정
# parser = argparse.ArgumentParser(description='receive the parameters')
# parser.add_argument('--topic_num', type = int, required = True)
# args = parser.parse_args()
# topic_num = args.topic_num

'''
경로 셋팅
'''
parent_dir = str(Path(os.getcwd()).parents[0])

'''
파라미터 로드
'''
topic_num = 1
my_task = 'test_eval'
my_dataset = 'topic-{}'.format(topic_num)
my_model = 'gpt2_small'
my_rl_lr = 0.0005
my_rl_bs = 256
my_rl_epoch = 20
my_decoding = 'stochastic'
my_prefix_len = 2
my_gen_len = 15
my_dropout = 'quantile'
my_dropout_rate = 0.95
my_test_prefix = "The issue focused on"

my_bs = 128

'''
타겟 정책 모델 로드
'''
# my_task == 'test_eval':
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

'''
행동 정책 모델  (objective 1)
'''
# 토크나이저 로드 및 모델 초기화
pretraind_config_dir = parent_dir + '/pretrained_weights' + '/' + my_model
pretrained_tokenizer_dir = pretraind_config_dir + '/tokenizer_left'
pretrained_weights_dir = pretraind_config_dir + '/model'
gpt_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_dir)
gpt_model = TFAutoModelForCausalLM.from_pretrained(pretrained_weights_dir)

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
보상 모델 로드 (objective 2)
'''
if 'topic' in my_dataset:
    num_labels = 4

# 토크나이저 로드 및 모델 초기화
pretraind_config_dir = parent_dir + '/pretrained_weights/bert'
pretrained_tokenizer_dir = pretraind_config_dir + '/tokenizer'
pretrained_weights_dir = pretraind_config_dir + '/model'
bert_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_dir)
bert_model = TFBertModel.from_pretrained(pretrained_weights_dir)
bert_model.resize_token_embeddings(len(bert_tokenizer))
bert_model = BERT_Classifier(bert_model, num_labels)

# 파인튜닝 가중치 로드
finetuned_weights_dir = parent_dir + '/weights' + '/' + my_dataset.split('-')[0] + '/bert'
my_model_ft_weights_dir = glob.glob(finetuned_weights_dir + '/*{}*'.format(my_bs))[0]
bert_model.load_weights(tf.train.latest_checkpoint(my_model_ft_weights_dir))

'''
평가용 데이터 로드
'''
# # 데이터 로드 경로 설정
# prep_data_path = parent_dir + '/prep_data' + '/' + my_dataset.split('-')[0] + '/' + my_model
test_gen_len = 30

'''
평가 수행
'''
# temperature_list = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
# for my_temperature in temperature_list:
#     print('temperature : {}'.format(my_temperature))

#     test_prefix = copy.deepcopy(my_test_prefix)
#     test_input = gpt_tokenizer(test_prefix, return_tensors='tf')['input_ids']
#     test_att = gpt_tokenizer(test_prefix, return_tensors='tf')['attention_mask']
#     if my_temperature != 0:
#         test_gen = target_model.generate(
#                                 test_input, 
#                                 attention_mask=test_att, 
#                                 max_new_tokens=test_gen_len,
#                                 pad_token_id=gpt_tokenizer.pad_token_id,
#                                 repetition_penalty=1.6,
#                                 do_sample=True, top_k=10, temperature=my_temperature)
#     else:
#         test_gen = target_model.generate(
#                                 test_input, 
#                                 attention_mask=test_att, 
#                                 max_new_tokens=test_gen_len,
#                                 pad_token_id=gpt_tokenizer.pad_token_id,
#                                 repetition_penalty=1.6,
#                                 do_sample=False)

'''
시드 셋팅
'''
start_time = time.time()

seed = 1234
np.random.seed(seed)
num_seeds = 50
my_seed_list = np.random.randint(1, 100000, size=num_seeds)
reward_score_list = []
bp_gen_ratio_list = []
for my_seed in my_seed_list:
    tf.random.set_seed(my_seed)
    np.random.seed(my_seed)
    random.seed(my_seed)

    '''
    생성 평가
    '''
    # bp_gen_len_list = [0, 2, 4, 8, 10, 15, 17, 20, 24, 28, 30]
    # bp_gen_len_list = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    bp_gen_len_list = np.arange(0, test_gen_len+1, 1)
    for bp_gen_len in bp_gen_len_list:
        print('--------------------------------------------------------------------------')

        if bp_gen_len == 0:
            print('bp_gen_len : {}'.format(bp_gen_len))
            bp_gen_ratio = bp_gen_len/test_gen_len
            print('bp_gen_ratio : {:.2f}'.format(bp_gen_ratio))
            test_prefix = copy.deepcopy(my_test_prefix)
            test_input = gpt_tokenizer(test_prefix, return_tensors='tf')['input_ids']
            test_att = gpt_tokenizer(test_prefix, return_tensors='tf')['attention_mask']
            print('test_prefix\n')
            print(test_prefix)
            print('\n')

            test_gen = target_model.generate(
                                    test_input, 
                                    attention_mask=test_att, 
                                    max_new_tokens=test_gen_len,
                                    pad_token_id=gpt_tokenizer.pad_token_id,
                                    repetition_penalty=1.6,
                                    do_sample=True, top_k=30, temperature=1.0)

            tp_gen_len = test_gen_len - bp_gen_len
            print('tp_gen_len : {}'.format(tp_gen_len))

        elif bp_gen_len == test_gen_len:
            print('bp_gen_len : {}'.format(bp_gen_len))
            bp_gen_ratio = bp_gen_len/test_gen_len
            print('bp_gen_ratio : {:.2f}'.format(bp_gen_ratio))
            test_prefix = copy.deepcopy(my_test_prefix)
            test_input = gpt_tokenizer(test_prefix, return_tensors='tf')['input_ids']
            test_att = gpt_tokenizer(test_prefix, return_tensors='tf')['attention_mask']
            print('test_prefix\n')
            print(test_prefix)
            print('\n')
            
            test_gen = gpt_model.generate(
                                    test_input, 
                                    attention_mask=test_att, 
                                    max_new_tokens=bp_gen_len,
                                    pad_token_id=gpt_tokenizer.pad_token_id,
                                    repetition_penalty=1.6,
                                    do_sample=False)

            bp_gen_len = test_gen_len - bp_gen_len
            print('bp_gen_len : {}'.format(bp_gen_len))

        else:
            print('bp_gen_len : {}'.format(bp_gen_len))
            bp_gen_ratio = bp_gen_len/test_gen_len
            print('bp_gen_ratio : {:.2f}'.format(bp_gen_ratio))
            test_prefix = copy.deepcopy(my_test_prefix)
            test_input = gpt_tokenizer(test_prefix, return_tensors='tf')['input_ids']
            test_att = gpt_tokenizer(test_prefix, return_tensors='tf')['attention_mask']

            test_bp_gen = gpt_model.generate(
                                    test_input, 
                                    attention_mask=test_att, 
                                    max_new_tokens=bp_gen_len,
                                    pad_token_id=gpt_tokenizer.pad_token_id,
                                    repetition_penalty=1.6,
                                    do_sample=False)

            test_bp_prefix = gpt_tokenizer.batch_decode(test_bp_gen)
            test_bp_input = gpt_tokenizer(test_bp_prefix, return_tensors='tf')['input_ids']
            test_bp_att = gpt_tokenizer(test_bp_prefix, return_tensors='tf')['attention_mask']
            print('test_(bp)_prefix\n')
            print(test_bp_prefix)
            print('\n')


            tp_gen_len = test_gen_len - bp_gen_len
            print('tp_gen_len : {}'.format(tp_gen_len))

            test_gen = target_model.generate(
                                    test_bp_input, 
                                    attention_mask=test_bp_att, 
                                    max_new_tokens=tp_gen_len,
                                    pad_token_id=gpt_tokenizer.pad_token_id,
                                    repetition_penalty=1.6,
                                    do_sample=True, top_k=30, temperature=1.0)

        test_gen_decoded = gpt_tokenizer.batch_decode(test_gen)
        print('test_gen_decoded\n')
        print(test_gen_decoded)
        print('\n')

        # 첫문장 추출
        # first_sentence = extract_first_sentence(test_gen_decoded[0])
        test_gen_sentences = extract_n_sentences(test_gen_decoded[0], n=1)

        # 우도 계산
        test_gen_sentences_gpt_encoded = gpt_tokenizer(test_gen_sentences, return_tensors='tf')
        gpt_test_gen_sentences = test_gen_sentences_gpt_encoded['input_ids']
        gpt_test_gen_masks = test_gen_sentences_gpt_encoded['attention_mask']
        outputs = gpt_model(gpt_test_gen_sentences, attention_mask = gpt_test_gen_masks, training = False)
        # lik_score = tf.reduce_mean(tf.nn.softmax(outputs.logits, axis = -1))
        lik_score = tf.reduce_mean(outputs.logits)
        print('lik_score : {:.5f}'.format(lik_score))
        # losses = loss_function(real=targets, pred=outputs.logits, mask=targets_mask)
        # accuracies = accuracy_function(real=targets, pred=outputs.logits, mask=targets_mask)


        # 보상 계산
        test_gen_sentences_bert_encoded = bert_tokenizer(test_gen_sentences, return_tensors='np', truncation=True, max_length=my_gen_len, padding=True)     # <bos> 포함
        bert_test_gen_sentences = test_gen_sentences_bert_encoded['input_ids']
        bert_test_gen_masks = test_gen_sentences_bert_encoded['attention_mask']        
        test_gen_class_logits = bert_model(bert_test_gen_sentences, attention_mask = bert_test_gen_masks, training = False)
        test_gen_class_logits = tf.squeeze(test_gen_class_logits)

        # 보상 점수 (topic-1의 경우)
        reward_score = tf.nn.softmax(test_gen_class_logits, axis = -1).numpy()[topic_num]
        print('reward_score : {:.5f}'.format(reward_score))
        print('\n')

        reward_score_list += [reward_score]
        bp_gen_ratio_list += [bp_gen_ratio]

unique_bp_gen_ratio_list = tf.cast(np.unique(bp_gen_ratio_list), dtype = tf.float32)[:, tf.newaxis]
per_bp_gen_ratio_mean_reward = tf.reduce_mean(tf.reshape(reward_score_list, shape = (num_seeds, -1)), axis = 0)[:, tf.newaxis]
per_bp_gen_ratio_mean_reward_with_bp_gen_ratio = tf.concat([unique_bp_gen_ratio_list, per_bp_gen_ratio_mean_reward], axis = 1).numpy()

plt.plot(per_bp_gen_ratio_mean_reward_with_bp_gen_ratio[:, 0], per_bp_gen_ratio_mean_reward_with_bp_gen_ratio[:, 1])
plt.xlabel('likelihood')
plt.ylabel('reward')
plt.show()

x = tf.cast(bp_gen_ratio_list, dtype=tf.float32)
y = tf.cast(reward_score_list, dtype=tf.float32)
plt.plot(x, y, 'k .')
plt.xlabel('likelihood')
plt.ylabel('reward')
plt.show()

end_time = time.time()

# 소요된 시간 계산
elapsed_time = end_time - start_time
print(f"코드 실행 시간: {elapsed_time}초")
# %%
