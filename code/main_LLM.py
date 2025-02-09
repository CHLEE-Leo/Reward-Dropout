# %%
import os
import json
from utils import initialize_setting, get_params, seed_everything, set_save_dir, createFolder
if __name__ == "__main__":

    '''
    환경설정
    '''
    # GPU setting
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    initialize_setting()

    '''
    입력받은 인자 가져오기
    '''
    args = get_params()

    '''
    시드 고정
    '''
    seed_everything(args.my_seed)
    # seed_everything(47)

    if args.task == 'ft':
        '''
        행동 정책 모델 사전훈련 인자
        '''
        my_task = args.task
        my_model = args.model
        my_dataset = args.dataset
        my_lr = args.lr
        my_bs = args.batch_size
        my_epoch = args.num_epoch
        my_patience = args.num_patience

        '''
        파라미터 정의
        '''
        kwargs = {
            'task' : my_task,
            'model' : my_model,
            'dataset' : my_dataset,
            'lr' : my_lr,
            'batch_size' : my_bs,
            'num_epoch' : my_epoch,
            'num_patience' : my_patience
        }

        '''
        파라미터 저장 경로 설정 및 저장
        '''
        SAVE_PARAM_DIR = set_save_dir(kwargs, folder='params', subfolder=my_dataset + '/' + my_model)
        with open(SAVE_PARAM_DIR + '/kwargs.json', 'w') as f:
            json.dump(kwargs, f)


        '''
        파일 실행
        '''
        if args.model in ['gpt2_small', 'gpt2_large', 'dialoGPT']:
            import finetune_GPT

        elif args.model == 'bert':
            import finetune_BERT


    elif args.task == 'rl':

        '''
        행동 정책 모델 인자
        '''
        my_task = args.task
        my_model = args.model
        my_dataset = args.dataset
        my_lr = args.lr
        my_bs = args.batch_size
        my_epoch = args.num_epoch
        my_patience = args.num_patience

        '''
        타겟 정책 모델 인자
        '''
        my_rl_bs = args.rl_batch_size
        my_rl_lr = args.rl_lr
        my_rl_epoch = args.rl_num_epoch
        my_decoding = args.decoding
        my_prefix_len = args.prefix_len
        my_gen_len = args.gen_len
        my_dropout = args.dropout
        my_dropout_rate = args.dropout_rate
        my_init_weight = args.init_weight
        my_reward = args.reward

        '''
        파라미터 정의
        '''
        kwargs = {
            'task' : my_task,
            'behavior_model' : my_model,
            'dataset' : my_dataset,
            'rl_lr' : my_rl_lr,
            'batch_size' : my_rl_bs,
            'num_epoch' : my_rl_epoch,
            'decoding' : my_decoding,
            'prefix_len' : my_prefix_len,
            'gen_len' : my_gen_len,
            'dropout' : my_dropout,
            'dropout_rate' : my_dropout_rate,
            'reward' : my_reward
        }

        '''
        파라미터 저장 경로 설정 및 저장
        '''
        # my_init_weight == None <==> my_model != gpt2_small_init_weight
        #                        <==> my_model == {gpt2_small, gpt2_large, opt, xglm, ..., etc}
        if my_init_weight == None:
            SAVE_PARAM_DIR = set_save_dir(kwargs, folder='params', subfolder=my_dataset + '/' + my_model)
            with open(SAVE_PARAM_DIR + '/kwargs.json', 'w') as f:
                json.dump(kwargs, f)

        # my_init_weight != None <==> my_model == gpt2_small_init_weight
        else:
            SAVE_PARAM_DIR = set_save_dir(kwargs, folder='params', subfolder=my_dataset + '/' + my_model + '=' + my_init_weight)
            with open(SAVE_PARAM_DIR + '/kwargs.json', 'w') as f:
                json.dump(kwargs, f)            

        '''
        파일 실행
        '''
        import reinforce_LLM
        # import reinforce_LLM_copy

    elif args.task == 'train_eval' or args.task == 'test_eval':

        '''
        행동 정책 모델 인자
        '''
        my_task = args.task
        my_model = args.model
        my_dataset = args.dataset

        '''
        타겟 정책 모델 인자
        '''
        my_rl_bs = args.rl_batch_size
        my_rl_lr = args.rl_lr
        my_rl_epoch = args.rl_num_epoch
        my_decoding = args.decoding
        my_prefix_len = args.prefix_len
        my_gen_len = args.gen_len
        my_dropout = args.dropout
        my_dropout_rate = args.dropout_rate
        my_reward = args.reward

        '''
        파라미터 정의
        '''
        kwargs = {
            'task' : my_task,
            'behavior_model' : my_model,
            'dataset' : my_dataset,
            'rl_lr' : my_rl_lr,
            'batch_size' : my_rl_bs,
            'num_epoch' : my_rl_epoch,
            'decoding' : my_decoding,
            'prefix_len' : my_prefix_len,
            'gen_len' : my_gen_len,
            'dropout' : my_dropout,
            'dropout_rate' : my_dropout_rate,
            'reward' : my_reward
        }


        '''
        파일 실행
        '''
        import evaluate_LLM