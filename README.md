# README.md
This repository provides the code, data, and results of the paper titled *"Reward Dropout Improves Control: Bi-objective Perpspective on Reinforced LM."*

<br/>

# Set-up

To use our code, first **clone this repository.** Then, **make your conda environment align with ours** by importing `settings.yaml` as below code:
```
# create a new Conda environment named [environment_name].
conda create -n [environment_name] python=3.10.8 -y

# activate the created Conda environment.
conda activate [environment_name]

# update the created Conda environment with `settings.yaml` configurations.
# Note that the name & prefix keys in `settings.yaml` must be edited first.
conda env update --file settings.yaml --prune
```
You have to wait for a while to create conda enviroment from `settings.yaml`. Be sure to use ``--prune`` option since it deletes all the dependencies not contained in `settings.yaml` and forces the created Conda environment to strictly follow the dependencies of `settings.yaml`.

Note that the values of **'name'** key in `settings.yaml` file must be editted based on your **\[environment_name\]** before running the code above. **name** key refers to the name of new conda environment to be created based on ``settings.yaml``.

![Alt text](image-3.png)

## *"Found conflicts!"* error
When running the codes above returns *"Found conflicts!"* error, please running the below code before ``conda env update --file settings.yaml --prune``. Then, the error may be resolved:
```
# force the channel_priority configuration of conda environment to false.
conda config --set channel_priority false
```

## In case you have conflict to set up Conda environment.
After importing `settings.yaml`, editting the **name** key, and resolving *'Found conflicts* error right way, if your environment still has conflict, then it is probably because a version of your NVIDIA driver or CUDA does not align with the versions of `tensorflow-gpu` and `transformers` dependencies in `settings.yaml`. So, please force the installation of specific version as below:
```
conda install -c conda-forge tensorflow-gpu=2.10.0 -y
conda install -c conda-forge tensorflow-probability -y
conda install -c huggingface transformers=4.28.1 -y
```

<br/>
<br/>

# Benchmark Experiments
You can *reproduce the benchmark experiments* described in the paper by running the below codes:

## 1) Preparing datasets
For benchmark experiments, first download the ``datasets.tar.gz``, ``pretrained_weights.tar.gz``, and ``weights.tar.gz`` in our [Google Drive](https://drive.google.com/drive/folders/1qOWjtDq8Ry2QRdKZvyGzSetFF-lggz7M?usp=drive_link). After then, store and unzip them at the corresponding folders named ``prep_data``, ``pretrained_weights`` and ``weights``, respectively, as below:
```
# After store the zipped files downloaded from Google drive, then unzip them into the corresponding folders.

# unzip datasets.tar.gz into prep_data folder.
cd ./prep_data              # move to prep_data directory.
tar -zxvf datasets.tar.gz   # unzip datasets.tar.gz file.

# unzip pretrained_weights.tar.gz into pretrained_weights folder.
cd ./pretrained_weights                 # move to pretrained_weights directory.
tar -zxvf pretrained_weights.tar.gz     # unzip pretrained_weights.tar.gz file.

# unzip weights.tar.gz into weights folder.
cd ./weights                    # move to weights directory.
tar -zxvf weights.tar.gz        # unzip datasets.tar.gz file.
```

For Google Drive capacity reasons, we upload ``weights.tar.gz`` that contains only the weights trained by the hyperparameter settings below:
- dataset : *sentiment*, *topic*, *emtion*
- decoding : *None*, *stochastic*
- dropout : *quantile*
- dropout rate : *0.95*

It will take some time to download, save and upzip ``weights.tar.gz`` to the weights folder.

Note that we provide ``pretrained_weights.tar.gz`` as zipped file for your convenience, but you can download it directly from Huggingface:
```
# import GPT2 tokenizer.
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2",
                                                bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, mask_token = MASK, sep_token = SEP,
                                                padding="max_length", truncation=True, padding_side='right')

# import GPT2 model.
gpt2_model = TFAutoModelForCausalLM.from_pretrained("gpt2")
```

## 2) Running our models
Completing preparing datasets, running the below code to reproduce the results of benchmark expriments. **This code reproduces Table 4 in the appendix of our paper.**
```
sh evaluate_LLM.sh
```
After running the code, check out the ``results`` directory. Then, you can see that sub-directories named by dataset-label pair (e.g., ``emotion-1``) were created, and there is ``gen_samples.txt`` file within each sub-directory. This text file contains the generated results.

- Since the ``weights.tar.gz`` file contains only the weights of limited hyperparameter settings (as we mentioned in *preparing datasets* step), ``sh evaluate_LLM.sh`` will **reproduce the Table 4 only for those limited settings.**

If you would like to put some customized prefixes (i.e., prompts) as inputs to our model, then open ``evaluate_LLM.sh`` file and enter your own prefixes in the ``--test_prefix`` argument as below

![Alt text](image-4.png)

Also, if you want to see the generated results without reward dropout, then revise ``--dropout=quantile`` to ``--dropout=None``, and ``--dropout_rate=0.95`` to ``--dropout_rate=0.0`` arguments in ``evaluate_LLM.sh`` file as below

![Alt text](image-5.png)

## 4) Total results of benchmark experiments
Under ``results`` directory, you can find ``final_result_table.csv`` file. It is a file that aggregates all the results of our benchmark experiments. **Table 1 in our paper was written based on this file.**

## 5) Human evaluation result
Under ``results`` directory, you can find ``real_fake_total_df.csv`` and ``final_result_real_label_total_df.csv`` files. They are the files that contain the result of human evaluation. **Figure 10 in the Appendix of our paper was plotted based on those files.** 

<br/>
<br/>

# Non-pretrained target LM vs. Pretrained target LM
Running the code below will reproduce Figure 1 (a) w/o initializing $\pi_{\theta}$ to $\beta_{\bar{{\theta}}}$.
```
python3 plot_train_reward_comparison_with_init_model_all.py --plot_model=gpt2_small_init_weight=uniform
```


Similarlly, running the code below ill reproduce Figure 1 (b) w/ initializing $\pi_{\theta}$ to $\beta_{\bar{{\theta}}}$.

```
python3 plot_train_reward_comparison_with_init_model_all.py --plot_model=gpt2_small
```

<br/>

# Performance Comparison across different (pretrained) LLMs
Running the code below will reproduce Figure 3 (a) and (b).
```
sh plot_train_reward_comparison_by_modell_all.sh
```

<br/>
<br/>

# Additional Experiments: 10-turn positioning game
You can *reproduce the 10-turn positioning game* described in the Appendix by running the below codes:

### 1) Generating the simulated data
You need to **generate the position change history of behavior agent.** 

To  do that, running the code below:
```
python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=6 --reward_alpha=1.25 --reward_beta=0.6
```
- *``--mode=gen`` option indicates that 10-turn positioning game is simulated to generate the episodes, i.e., trajectories, of behavior agent.*
- *``--num_epi=5000`` option configures the total number of simulated episodes to be 5000.*
- *``--min_reward_action=6`` option configures the 10-turn positioning game where rewards are only collectable at positions from 6 to 10. Note that the position is defined as an integer number between 0 to 10.*
- *``--reward_alpha=alpha`` and ``--reward_beta=beta`` options determines the shape of reward distribution.* If setting ``alpha=1.25`` and ``beta=0.6``, then you can reproduce the reward distribution of Figure 1b in the paper.

Since the 10-turn positioning game forces the length of episode to be 10, the total number of position changes for the behavior agent is $50000 (= 10 \times 5000)$. 

In addition, as described in the paper, we tested 6 cases by different parameters of behavior policy, i.e., different combinations of $\mu$ and $\sigma$, so the total size of data increases up to $300000 (= 50000 \times 6$).

### 2) Training a target agent on the simulated data
You need to **train a target agent according to the position change history of behavior agent.**

To  do that, running the code below:
```
CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=0 --min_reward_action=6 --reward_alpha=1.25 --reward_beta=0.6

CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=1 --min_reward_action=6 --reward_alpha=1.25 --reward_beta=0.6

CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=2 --min_reward_action=6 --reward_alpha=1.25 --reward_beta=0.6

CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=3 --min_reward_action=6 --reward_alpha=1.25 --reward_beta=0.6

CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=4 --min_reward_action=6 --reward_alpha=1.25 --reward_beta=0.6

CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=6 --reward_alpha=1.25 --reward_beta=0.6
```
or
```
sh train_pos_game_agent.sh
```

- *``--batch_size=1`` option sets the batch size of the target agent to be 1e-05.*
- *``--lr=1e-05`` option sets the learning rate of the target agent to be 1e-05.*
- *``--num_epoch=1`` option sets the number of training epoch for the target agent to be 1.*
- *``--case=0`` option determines the simulated data of specific behavior agent to be learned by the target agent. In other words, it specifies the pre-defined combination of* $\mu$ *and* $\sigma$ *of the behavioral agent that the target agent will learn from. Since only 6 cases are considered in the paper, the 0, 1, 2, 3, 4, and 5 arguments are allowed for this option.*
- *``--min_reward_action=6`` option specifies that the target agent will be trained on the simulated data where rewards are only collectable at positions from 6 to 10.*
- *``--reward_alpha`` and ``--reward_beta`` options specify that the target agent will be trained on the simulated data where rewards are distributed according to Beta(alpha, beta) distribution.*

Note that you can change ``CUDA_VISIBLE_DEVICES=0`` command depending on the GPU availability in your server.

### 3) Evaluating target agent's policy
After training, evaluate the target agent's policy by initializing its position to a random state and running it to play a 10-turn positioning game. 

To  do that, running the code below:
```
CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=0 --min_reward_action=6 --reward_alpha=1.25 --reward_beta=0.6

CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=1 --min_reward_action=6 --reward_alpha=1.25 --reward_beta=0.6

CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=2 --min_reward_action=6 --reward_alpha=1.25 --reward_beta=0.6

CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=3 --min_reward_action=6 --reward_alpha=1.25 --reward_beta=0.6

CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=4 --min_reward_action=6 --reward_alpha=1.25 --reward_beta=0.6

CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=6 --reward_alpha=1.25 --reward_beta=0.6

```
or
```
sh evaluate_pos_game_agent.sh
```

### 4) Plotting the results
Now, you need to draw the resulting plots, i.e., the action or position distribution of the target agent.

To  do that, running the code below:
```
CUDA_VISIBLE_DEVICES=0 python3 pos_game_data.py --mode=test --num_epi=5000 --batch_size=1 --lr=1e-05 --num_epoch=1 --min_reward_action=6 --reward_alpha=1.25 --reward_beta=0.6
```
Running this code will generate a file named ``figure1_1e-05_1_5_mra=6_alpha=1.25_beta=0.6.pdf`` that **reproduces Figure 11b in our paper.**

### 5) Conducting additional analysis
We have analyzed the dynamics of target agent given the behavior agent (behavior policy) is fixed while the different reward distribution was set up according to the power of 2 at $i \in [1, 10]$.

To visualize the result of analysis, running the code below:
```
sh plot_additional_analysis.sh
```
Running this code will generate a file named ``4col_figure1_1e-05_1_5_mra=1_alpha=15.0_beta=15.0.pdf`` that **reproduces the second row of Figure 12 in our paper.**

or 

you may manipulate the reward distribution by customizing your own alpha and beta as below:
```
# Generate a simulated data of 10-turn positioning game.

python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_alpha=1.0 --reward_beta=1.0
python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_alpha=5.0 --reward_beta=5.0
python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_alpha=10.0 --reward_beta=10.0
python3 pos_game_data.py --mode=gen --num_epi=5000 --min_reward_action=1 --reward_alpha=15.0 --reward_beta=15.0

# Train agent in different reward distributions.
# Note that case=5 argument in the below code refers to the behavior policy at case 2 in Figure 1b of our paper.

CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_alpha=1.0 --reward_beta=1.0
CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_alpha=5.0 --reward_beta=5.0
CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_alpha=10.0 --reward_beta=10.0
CUDA_VISIBLE_DEVICES=0 python3 train_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_alpha=15.0 --reward_beta=15.0

# Evaluate the trained agent starting from a random initial position.

CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_alpha=1.0 --reward_beta=1.0
CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_alpha=5.0 --reward_beta=5.0
CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_alpha=10.0 --reward_beta=10.0
CUDA_VISIBLE_DEVICES=0 python3 evaluate_pos_game_agent.py --batch_size=1 --lr=1e-05 --num_epoch=1 --case=5 --min_reward_action=1 --reward_alpha=15.0 --reward_beta=15.0

# Plot the result of 10-turn positioning game.
CUDA_VISIBLE_DEVICES=0 python3 plot_additional_analysis.py --num_epi=5000 --batch_size=1 --lr=1e-05 --num_epoch=1 --min_reward_action=1
```

<br/>
<br/>
