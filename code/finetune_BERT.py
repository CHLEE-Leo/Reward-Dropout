# %%
import os
from pathlib import Path
import argparse
import json
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
import gc
import time
import tqdm
import glob


from tensorflow.keras.utils import Progbar
from load_pretrained_LLM import model
from train_function_BERT import train_step, test_step
from utils import *

'''
파라미터 로드
'''
args = get_params()
my_task = args.task
my_dataset = args.dataset
my_model = args.model
my_lr = args.lr
my_bs = args.batch_size
my_epoch = args.num_epoch
my_patience = args.num_patience

parent_dir = str(Path(os.getcwd()).parents[0])
PARAM_DIR = parent_dir + '/params' + '/' + str(my_dataset) + '/' + str(my_model)
PARAM_DIR = glob.glob(PARAM_DIR + '/*{}**{}**{}**{}**{}*'.format(my_task, my_lr, my_bs, my_epoch, my_patience))[0]
with open(PARAM_DIR + '/kwargs.json', 'r') as f:
    kwargs = json.load(f)

'''
데이터 로드
'''
# 데이터 로드 경로 설정
prep_data_path = parent_dir + '/prep_data' + '/' + my_dataset + '/' + my_model

'''
훈련용 데이터 셋팅
'''
# 훈련용 인풋 시퀀스, 시퀀스의 어텐션 마스크, 라벨 로드
train_input_x = np.load(prep_data_path + '/train_input_x.npy')
train_input_att = np.load(prep_data_path + '/train_input_att.npy')
train_target_y_label = np.load(prep_data_path + '/train_target_y.npy')


'''
검증용 데이터 셋팅
'''
# 검증용 인풋 시퀀스, 시퀀스의 어텐션 마스크, 라벨 로드
test_input_x = np.load(prep_data_path + '/test_input_x.npy')
test_input_att = np.load(prep_data_path + '/test_input_att.npy')
test_target_y_label = np.load(prep_data_path + '/test_target_y.npy')


'''
훈련/검증 데이터셋 구축
'''
with tf.device("/cpu:0"):

    # 훈련 셋
    train_seq = (train_input_x, train_input_att)    
    
    train_dat = tf.data.Dataset.from_tensor_slices((train_seq, train_target_y_label)).shuffle(buffer_size = train_input_x.shape[0], reshuffle_each_iteration = False)
    train_batch = train_dat.batch(batch_size=my_bs, drop_remainder=True)

    # 검증 셋
    test_seq = (test_input_x, test_input_att)    

    test_dat = tf.data.Dataset.from_tensor_slices((test_seq, test_target_y_label)).shuffle(buffer_size = test_input_x.shape[0], reshuffle_each_iteration = False)
    test_batch = test_dat.batch(batch_size=my_bs, drop_remainder=True)


''''
훈련 과정 수행
'''
# 학습 가중치 및 결과 저장경로 생성
SAVE_WEIGHT_DIR = set_save_dir(kwargs, folder='weights', subfolder=my_dataset + '/' + my_model)
SAVE_RESULT_DIR = set_save_dir(kwargs, folder='results', subfolder=my_dataset + '/' + my_model)

# 훈련 메트릭
metrics_names = [str(my_model) + '_loss', str(my_model) + '_acc']
train_pre_acc = train_pre_loss = 0.0
test_pre_acc = max_test_pre_acc = test_pre_loss = 0.0 
k = my_patience
patience_list = []

train_loss_history = []
test_loss_history = []
train_acc_history = []
test_acc_history = []

# 훈련 루프
total_start_time = time.time()

for epoch in range(my_epoch):

    # 훈련 시간 및 진행상황 측정
    start_time = time.time()
    print("\nepoch {}/{}".format(epoch + 1, my_epoch))
    pb_i = Progbar(len(train_batch), stateful_metrics = metrics_names)

    # 매 epoch마다 훈련/검증 정확도 및 손실 초기화
    train_cumul_acc = test_cumul_acc = 0
    train_cumul_loss = test_cumul_loss = 0

    # 훈련 배치 루프
    for idx, (train_seq, train_labels) in enumerate(train_batch):        

        # 시퀀스와 어텐션 마스크 분리
        train_inputs, train_inputs_mask = train_seq

        # train_function_LLM.py의 train_step() 을 통해 훈련 수행
        train_loss, train_acc = train_step((train_inputs, train_inputs_mask, train_labels), model)

        # 배치별 정확도 및 손실 누계
        train_cumul_acc += train_acc.numpy()
        train_cumul_loss += train_loss.numpy()

        # 메트릭 값 업데이트
        metric_values = [(str(my_model) + '_loss', train_loss), (str(my_model) + '_acc', train_acc)]
        pb_i.update(idx+1, values = metric_values)


    # 전체 평균 정확도 및 손실 (훈련셋)
    train_mean_acc = train_cumul_acc/(idx + 1)
    train_mean_loss = train_cumul_loss/(idx + 1)

    # 훈련 성능 출력
    train_acc_delta = train_mean_acc - train_pre_acc
    print('train_mean_loss : {}, train_mean_acc : {}, train_pre_acc : {}, train_acc_delta : {}'.format(train_mean_loss, train_mean_acc, train_pre_acc, train_acc_delta))
    train_pre_acc = train_mean_acc

    # 검증 배치 루프
    for idx, (test_seq, test_labels) in enumerate(test_batch):  

        # 시퀀스와 어텐션 마스크 분리
        test_inputs, test_inputs_mask = test_seq

        # test_function_LLM.py의 test_step() 을 통해 훈련 수행
        test_loss, test_acc = test_step((test_inputs, test_inputs_mask, test_labels), model)

        # 배치별 정확도 및 손실 누계
        test_cumul_acc += test_acc.numpy()
        test_cumul_loss += test_loss.numpy()

    # 전체 평균 정확도 및 손실 (검증셋)
    test_mean_loss = test_cumul_loss/(idx + 1)
    test_mean_acc = test_cumul_acc/(idx + 1)

    # 검증 성능 출력
    test_acc_delta = test_mean_acc - test_pre_acc
    print('test_mean_loss : {}, test_mean_acc : {}, test_pre_acc : {}, test_acc_delta : {}, max_test_pre_acc : {}'.format(test_mean_loss, test_mean_acc, test_pre_acc, test_acc_delta, max_test_pre_acc))
    test_pre_acc = test_mean_acc

    # 가중치 저장 조건
    '''
    test set에 대해서 이전 epoch에서 집계된 최고 성능치보다 현재 epoch의 성능치가 개선될 경우 저장
    '''
    max_test_acc_delta = test_mean_acc - max_test_pre_acc
    if max_test_acc_delta > 0.0:

        # 현 정확도가 가장 높았던 이전 정확도보다 개선됐을 경우에만 이전 정확도 값 업데이트
        max_test_pre_acc = test_mean_acc

        # 현 정확도가 가장 높았던 이전 정확도보다 개선됐을 경우에만 가중치 저장
        model.save_weights(SAVE_WEIGHT_DIR + '/weights.ckpt')

    # 훈련 / 검증 셋 손실 히스토리 저장
    train_loss_history += [train_mean_loss]
    test_loss_history += [test_mean_loss]
    loss_history_pd = pd.DataFrame(zip(train_loss_history, test_loss_history), columns = ['train_loss', 'test_loss'])
    loss_history_pd.to_csv(SAVE_RESULT_DIR + '/loss_history.csv', index_label = 'epoch')

    # 훈련 / 검증 셋 정확도 히스토리 저장
    train_acc_history += [train_mean_acc]
    test_acc_history += [test_mean_acc]
    acc_history_pd = pd.DataFrame(zip(train_acc_history, test_acc_history), columns = ['train_acc', 'test_acc'])
    acc_history_pd.to_csv(SAVE_RESULT_DIR + '/acc_history.csv', index_label = 'epoch')

    # 학습 중단 조건
    '''
    test_batch에 대해서 이전 k-epoch 동안 성능이 연속으로 저하되거나 훈련/검증 정확도 지표가 모두 0.95를 넘을 경우 경우 중단
    - my_patience > 0 일 경우 종료조건이 활성화됨
    - my_patience = 0 일 경우, my_epoch 만큼 돌아감.
    '''
    if my_patience > 0:
        if len(patience_list) < k:
            patience_list += [test_acc_delta]
        else:
            del patience_list[0]
            patience_list += [test_acc_delta]            
        print('patience_list :', patience_list)
        if len(np.where(np.array(patience_list) < 0)[0]) == k or (train_mean_acc + test_mean_acc) > (2 * 0.95):
            break;

    end_time = time.time()
    cur_sec = (end_time - start_time)%60
    cur_min = ((end_time - start_time)//60)%60
    cur_hr = ((end_time - start_time)//60)//60
    total_sec = (end_time - total_start_time)%60
    total_min = ((end_time - total_start_time)//60)%60
    total_hr = ((end_time - total_start_time)//60)//60
    print("elapsed time : {:.0f} hr, {:.0f} min, {:.2f} sec".format(cur_hr, cur_min, cur_sec))
    print("total elapsed time : {:.0f} hr, {:.0f} min, {:.2f} sec".format(total_hr, total_min, total_sec))
