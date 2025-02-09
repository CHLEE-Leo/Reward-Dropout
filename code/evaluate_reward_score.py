# %%
import os
from pathlib import Path
import sys
import argparse
import json
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Progbar
import gc
import time
import tqdm
import glob
import matplotlib.pyplot as plt

# from utils import get_params, set_save_dir, indice_pad_in_prefix, remove_pad_in_prefix_case, right_pad_after_eos_token, truncate_datasize_by_ratio, get_truncated_data, extract_first_sentence, extract_n_sentences, seed_everything, get_first_n_words
from utils import get_params, set_save_dir, extract_n_sentences, seed_everything, get_first_n_words

'''
파라미터 로드
'''
args = get_params()
my_task = args.task
my_dataset = args.dataset
my_model = 'gpt2_small'
my_rl_lr = 0.0005
my_rl_bs = 256
my_rl_epoch = 20
my_decoding = args.decoding
my_prefix_len = 2
my_gen_len = 15
my_dropout = args.dropout
my_dropout_rate = args.dropout_rate
my_test_prefix = args.test_prefix
# my_task = 'train_reward'
# my_dataset = 'sentiment-1'
# my_decoding = 'stochastic'
# my_dropout = 'quantile'
# my_dropout_rate = 0.2

parent_dir = str(Path(os.getcwd()).parents[0])
PARAM_DIR = parent_dir + '/params' + '/' + str(my_dataset) + '/' + str(my_model)
PARAM_DIR = glob.glob(PARAM_DIR + '/*{}**{}**{}*'.format(my_decoding, my_dropout, my_dropout_rate))[0]
with open(PARAM_DIR + '/kwargs.json', 'r') as f:
    kwargs = json.load(f)

'''
데이터 셋팅
'''
# 훈련 보상 플롯 
if my_task == 'train_eval':
    # '''
    # 훈련 보상 로드
    # '''
    # SAVE_RESULT_DIR = set_save_dir(kwargs, folder='results', subfolder=my_dataset + '/' + my_model)
    # train_reward_df = pd.read_csv(SAVE_RESULT_DIR + '/reward_history.csv', index_col=0)
    # min_val = round(np.arange(0.5, 1.0, 0.05)[np.where(np.array(np.arange(0.5, 1.0, 0.05) < np.min(train_reward_df)) == False)[0][0]-1], 2)
    # max_val = round(np.arange(0.5, 1.0, 0.05)[np.argmax(np.arange(0.5, 1.0, 0.05) > np.max(train_reward_df))], 2)
    # plt.figure(figsize=(3, 2))
    # plt.plot(train_reward_df)
    # plt.ylabel('reward')
    # plt.yticks(ticks=np.round(np.arange(min_val, max_val, 0.01), 2), labels=np.round(np.arange(min_val, max_val, 0.01), 2))
    # plt.xlabel('episode (= epoch)')
    # plt.title('dataset : {} / decoding : {} \n dropout : {} / dropout_rate : {}'.format(my_dataset, my_decoding, my_dropout, my_dropout_rate), fontsize=10)

    my_decoding_list = ['greedy', 'stochastic', 'top-k']
    my_dropout_list = ['None', 'random', 'quantile']
    my_dropout_rate_list = [0.8, 0.9, 0.95]
    my_dataset_list = [
                        ['sentiment-0', 'sentiment-1'],
                        ['politeness-0', 'politeness-1'],
                        ['toxicity-0', 'toxicity-1'],
                        ['emotion-1', 'emotion-2', 'emotion-3', 'emotion-4', 'emotion-5', 'emotion-6'],
                        ['topic-0', 'topic-1', 'topic-2', 'topic-3']
                       ]
    
    for idx4, my_decoding in enumerate(my_decoding_list):

        for idx3, my_dropout in enumerate(my_dropout_list):

            for idx2, my_dropout_rate in enumerate(my_dropout_rate_list):

                for idx, data_level in enumerate(my_dataset_list):
                    num_classes = 0
                    total_reward = 0

                    if my_dropout == 'None':
                        my_dropout_rate = 0.0

                    for class_level in data_level:
                        num_classes += 1
                        data_dir = parent_dir + '/results/{}/gpt2_small/rl_gpt2_small_{}_{}_{}_{}_{}_{}_{}_{}_{}/reward_history.csv'.format(class_level, class_level, my_rl_lr, my_rl_bs, my_rl_epoch, my_decoding, my_prefix_len, my_gen_len, my_dropout, my_dropout_rate)
                        reward_history_df = pd.read_csv(data_dir, index_col=0)
                        final_reward = reward_history_df['train_reward'][-1:]
                        total_reward += np.array(final_reward)

                    target_decoding = np.array([my_decoding])
                    print('target_decoding : {}'.format(target_decoding))

                    target_dropout = np.array([my_dropout])
                    print('target_dropout : {}'.format(target_dropout))

                    target_dropout_rate = np.array([my_dropout_rate])
                    print('target_dropout_rate : {}'.format(target_dropout_rate))

                    target_class = np.array([class_level.split('-')[0]])
                    print('target_class : {}'.format(target_class))

                    mean_reward = total_reward/num_classes
                    print('mean_reward : {}'.format(mean_reward))

                    # 보상정보 요약
                    reward_summary = pd.DataFrame([target_decoding, target_dropout, target_dropout_rate, target_class, mean_reward]).T

                    if idx == 0:
                        total_reward_summary = copy.deepcopy(reward_summary)
                    else:
                        total_reward_summary = pd.concat([total_reward_summary, reward_summary], axis = 0)

                    # if my_dropout == 'None':
                    #     break;

                if idx2 == 0:
                    total_reward_summary2 = copy.deepcopy(total_reward_summary)
                else:
                    total_reward_summary2 = pd.concat([total_reward_summary2, total_reward_summary], axis = 0)

                # if my_dropout == 'None':
                #     break;

            if idx3 == 0:
                total_reward_summary3 = copy.deepcopy(total_reward_summary2)
            else:
                total_reward_summary3 = pd.concat([total_reward_summary3, total_reward_summary2], axis = 0)

        if idx4 == 0:
            total_reward_summary4 = copy.deepcopy(total_reward_summary3)
        else:
            total_reward_summary4 = pd.concat([total_reward_summary4, total_reward_summary3], axis = 0)

    total_reward_summary4.columns = ['decoding', 'dropout', 'rate', 'dataset', 'mean_reward']
    final_result_table = total_reward_summary4.groupby(['decoding', 'dropout', 'rate', 'dataset']).aggregate('mean')
    
    save_dir = parent_dir + '/results'
    final_result_table.to_csv(save_dir + '/final_result_table.csv')

# 평가 성능 검증
elif my_task == 'test_eval':

    '''
    타겟 모델 로드
    '''
    from load_pretrained_LLM import gpt_tokenizer, target_model, bert_tokenizer, bert_model

    '''
    평가용 데이터 로드
    '''
    # # 데이터 로드 경로 설정
    # prep_data_path = parent_dir + '/prep_data' + '/' + my_dataset.split('-')[0] + '/' + my_model
    test_gen_len = 30

    '''
    평가 수행
    '''
    test_prefix = copy.deepcopy(my_test_prefix)
    test_input = gpt_tokenizer(test_prefix, return_tensors='tf')['input_ids']
    test_att = gpt_tokenizer(test_prefix, return_tensors='tf')['attention_mask']
    test_gen = target_model.generate(
                            test_input, 
                            attention_mask=test_att, 
                            max_new_tokens=test_gen_len,
                            pad_token_id=gpt_tokenizer.pad_token_id,
                            repetition_penalty=1.6,
                            # do_sample=True, temperature=0.7)
                            do_sample=True, top_k=10, temperature=0.3)
                            # do_sample=False)
    test_gen_decoded = gpt_tokenizer.batch_decode(test_gen)

    # 첫문장 추출
    # first_sentence = extract_first_sentence(test_gen_decoded[0])
    test_gen_sentences = extract_n_sentences(test_gen_decoded[0], n=1)

    # 보상 계산
    test_gen_sentences_bert_encoded = bert_tokenizer(test_gen_sentences, return_tensors='np', truncation=True, max_length=my_gen_len, padding=True)     # <bos> 포함
    bert_test_gen_sentences = test_gen_sentences_bert_encoded['input_ids']
    bert_test_gen_masks = test_gen_sentences_bert_encoded['attention_mask']        
    test_gen_pred_logits = bert_model(bert_test_gen_sentences, attention_mask = bert_test_gen_masks, training = False)

    '''
    생성결과 저장
    '''
    SAVE_RESULT_DIR = set_save_dir(kwargs, folder='results', subfolder=my_dataset + '/' + my_model)
    with open(SAVE_RESULT_DIR + '/gen_sample.txt', 'a') as f:
        # f.write('\n' + test_gen_decoded[0].split('.')[0] + '.')
        f.write('\n' + test_gen_sentences)

elif my_task == 'human_eval':
    '''
    타겟 모델 로드
    '''
    import glob
    from transformers import AutoTokenizer, TFAutoModelForCausalLM

    my_model = 'gpt2_small'
    my_decoding = 'stochastic'
    my_dropout = 'quantile'
    my_dropout_rate = 0.95
    my_dataset_list = [
                        ['sentiment-0', 'sentiment-1'], 
                        ['politeness-0', 'politeness-1'], 
                        ['topic-0', 'topic-1', 'topic-2', 'topic-3']
                    ]
    seed_everything(47)

    # '''
    # 세팅 1 (: real vs. fake 맞추기)
    # '''
    # # 각 데이터 세트별 > 각 라벨별 > 문장 뽑기 (4개)
    # # 단, emotion 데이터는 제외.
    # test_gen_len = 30
    # for idx1, dataset_level in enumerate(my_dataset_list):
    #     for idx2, label_level in enumerate(dataset_level):

    #         # 타겟 데이터셋 정의
    #         my_dataset = copy.deepcopy(label_level)

    #         # 타겟 데이터 셋 로드
    #         target_dataset_name = label_level.split('-')[0]
    #         target_file_dir = parent_dir + '/data/{}/train.csv'.format(target_dataset_name)
    #         target_dataset = pd.read_csv(target_file_dir, index_col=0)

    #         # 텍스트 데이터 및 라벨 데이터 구축
    #         text_data = target_dataset.iloc[:, 0]
    #         label_data = target_dataset.iloc[:, 1].astype('category').cat.codes
            
    #         # 임의의 4개 샘플의 인덱스 값을 비복원 추출
    #         sample_idx_vector = np.arange(target_dataset.shape[0])
    #         random_sample_idx = np.random.choice(sample_idx_vector, size=4, replace=False)
    #         real_text = text_data.iloc[random_sample_idx]
    #         real_fake_label = [1, 0]

    #         '''
    #         훈련된 모델 로드
    #         '''
    #         # 토크나이저 로드 및 모델 초기화
    #         pretraind_config_dir = parent_dir + '/pretrained_weights' + '/' + my_model
    #         pretrained_tokenizer_dir = pretraind_config_dir + '/tokenizer_right'
    #         pretrained_weights_dir = pretraind_config_dir + '/model'
    #         gpt_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_dir)
    #         target_model = TFAutoModelForCausalLM.from_pretrained(pretrained_weights_dir)

    #         # 훈련 가중치 주소 정의
    #         reinforced_weights_dir = parent_dir + '/weights' + '/' + my_dataset + '/' + my_model
    #         my_model_rl_weights_dir = glob.glob(reinforced_weights_dir + '/*{}**{}**{}*'.format(my_decoding, my_dropout, my_dropout_rate))[0]
    #         # print('my_model_rl_weights_dir :', my_model_rl_weights_dir)

    #         # 훈련 가중치 로드
    #         target_model.load_weights(tf.train.latest_checkpoint(my_model_rl_weights_dir))

    #         for idx3 in range(len(random_sample_idx)):

    #             # 실제 문장 로드 (생성된 전체 문장의 두번째 문장까지만 필터링)
    #             # real_text_extracted = extract_first_sentence(list(real_text)[i])
    #             real_text_extracted = extract_n_sentences(list(real_text)[idx3], n=1)
    #             print('real_text_extracted :', real_text_extracted)

    #             # 실제 문장의 일부 (prefix) 추출
    #             real_text_prefix = get_first_n_words(real_text_extracted, n=4)
    #             print('real_text_prefix :', real_text_prefix)

    #             # prefix로부터 문장 생성
    #             test_prefix = copy.deepcopy(real_text_prefix)
    #             test_input = gpt_tokenizer(test_prefix, return_tensors='tf')['input_ids']
    #             test_att = gpt_tokenizer(test_prefix, return_tensors='tf')['attention_mask']
    #             test_gen = target_model.generate(
    #                                     test_input, 
    #                                     attention_mask=test_att, 
    #                                     max_new_tokens=test_gen_len,
    #                                     pad_token_id=gpt_tokenizer.pad_token_id,
    #                                     repetition_penalty=1.6,
    #                                     do_sample=True, top_k=10, temperature=0.3)
    #                                     # do_sample=False)
    #             test_gen_decoded = gpt_tokenizer.batch_decode(test_gen)
    #             test_gen_decoded = extract_n_sentences(test_gen_decoded[0], n=1)
    #             print('gen_text :', test_gen_decoded)
    #             print('\n')
                
    #             if idx1 == 0 and idx2 == 0 and idx3 == 0:
    #                 real_fake_text = [real_text_extracted, test_gen_decoded]
    #                 real_fake_result = list([real_fake_text, real_fake_label])
    #                 real_fake_df = pd.DataFrame(real_fake_result).T.sample(frac=1)
    #                 real_fake_total_df = copy.deepcopy(real_fake_df)
    #             else:
    #                 real_fake_text = [real_text_extracted, test_gen_decoded]
    #                 real_fake_result = list([real_fake_text, real_fake_label])
    #                 real_fake_df = pd.DataFrame(real_fake_result).T.sample(frac=1)
    #                 real_fake_total_df = pd.concat([real_fake_total_df, real_fake_df], axis = 0)

    # # 전체 데이터셋 저장
    # real_fake_total_df.columns = ['text', 'label']
    # real_fake_total_df.to_csv(parent_dir + '/results/real_fake_total_df.csv', index=False)   # row-shuffling

    '''
    세팅 2 (: control attribute 맞추기)
    '''
    # 각 데이터 세트별 > 각 라벨별 > 문장 뽑기 (4개)
    # 단, emotion 데이터는 제외.
    test_gen_len = 50
    for idx1, dataset_level in enumerate(my_dataset_list):
        for idx2, label_level in enumerate(dataset_level):

            # 타겟 데이터셋 정의
            my_dataset = copy.deepcopy(label_level)

            # 타겟 데이터 셋 로드
            target_dataset_name = label_level.split('-')[0]
            target_file_dir = parent_dir + '/data/{}/train.csv'.format(target_dataset_name)
            target_dataset = pd.read_csv(target_file_dir, index_col=0)

            # 텍스트 데이터 및 라벨 데이터 구축
            text_data = target_dataset.iloc[:, 0]
            label_data = target_dataset.iloc[:, 1].astype('category').cat.codes
            
            # 임의의 4개 샘플의 인덱스 값을 비복원 추출
            sample_idx_vector = np.arange(target_dataset.shape[0])
            random_sample_idx = np.random.choice(sample_idx_vector, size=4, replace=False)
            real_text = text_data.iloc[random_sample_idx]
            # real_label = label_data.iloc[random_sample_idx]
            target_label = label_level.split('-')[-1]

            '''
            훈련된 모델 로드
            '''
            # 토크나이저 로드 및 모델 초기화
            pretraind_config_dir = parent_dir + '/pretrained_weights' + '/' + my_model
            pretrained_tokenizer_dir = pretraind_config_dir + '/tokenizer_right'
            pretrained_weights_dir = pretraind_config_dir + '/model'
            gpt_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_dir)
            target_model = TFAutoModelForCausalLM.from_pretrained(pretrained_weights_dir)

            # 훈련 가중치 주소 정의
            reinforced_weights_dir = parent_dir + '/weights' + '/' + my_dataset + '/' + my_model
            my_model_rl_weights_dir = glob.glob(reinforced_weights_dir + '/*{}**{}**{}*'.format(my_decoding, my_dropout, my_dropout_rate))[0]
            # print('my_model_rl_weights_dir :', my_model_rl_weights_dir)

            # 훈련 가중치 로드
            target_model.load_weights(tf.train.latest_checkpoint(my_model_rl_weights_dir))

            for idx3 in range(len(random_sample_idx)):
                # 실제 문장 로드 (생성된 전체 문장의 두번째 문장까지만 필터링)
                # real_text_extracted = extract_first_sentence(list(real_text)[i])
                real_text_extracted = extract_n_sentences(list(real_text)[idx3], n=2)
                print('real_text_extracted :', real_text_extracted)

                # 실제 문장의 일부 (prefix) 추출
                real_text_prefix = get_first_n_words(real_text_extracted, n=2)
                print('real_text_prefix :', real_text_prefix)

                # prefix로부터 문장 생성
                test_prefix = copy.deepcopy(real_text_prefix)
                test_input = gpt_tokenizer(test_prefix, return_tensors='tf')['input_ids']
                test_att = gpt_tokenizer(test_prefix, return_tensors='tf')['attention_mask']
                test_gen = target_model.generate(
                                        test_input, 
                                        attention_mask=test_att, 
                                        max_new_tokens=test_gen_len,
                                        pad_token_id=gpt_tokenizer.pad_token_id,
                                        repetition_penalty=1.6,
                                        do_sample=True, top_k=10, temperature=0.3)
                                        # do_sample=False)
                test_gen_decoded = gpt_tokenizer.batch_decode(test_gen)
                test_gen_decoded = extract_n_sentences(test_gen_decoded[0], n=2)
                print('gen_text :', test_gen_decoded)
                print('\n')

                if idx1 == 0 and idx2 == 0 and idx3 == 0:
                    fake_text = [test_gen_decoded]
                    target_attribute = [target_label]
                    fake_result_real_label = list([fake_text, target_attribute])
                    fake_result_real_label_df = pd.DataFrame(fake_result_real_label).T
                    fake_result_real_label_df['dataset'] = target_dataset_name
                    fake_result_real_label_total_df = copy.deepcopy(fake_result_real_label_df)
                else:
                    fake_text = [test_gen_decoded]
                    target_attribute = [target_label]
                    fake_result_real_label = list([fake_text, target_attribute])
                    fake_result_real_label_df = pd.DataFrame(fake_result_real_label).T
                    fake_result_real_label_df['dataset'] = target_dataset_name
                    fake_result_real_label_total_df = pd.concat([fake_result_real_label_total_df, fake_result_real_label_df], axis = 0)

            # gen_prefix_text = random_sample_idx[2:]

    # 전체 데이터셋 저장
    fake_result_real_label_total_df.columns = ['text', 'label', 'dataset']
    fake_result_real_label_total_df.to_csv(parent_dir + '/results/fake_result_real_label_total_df.csv', index=False)
