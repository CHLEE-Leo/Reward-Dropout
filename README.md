# README.md
This repository provides the codes, data, and results of the paper titled *"Towards Pareto-Efficient RLHF: Paying Attention to a Few High-Reward Samples with Reward Dropout"*

Note that the codes and results in this repository do not fully cover the contents of the paper, but it would be enough to get an idea of the paper. 

<br/>

# How to run codes.
After setting your environment, if you want to train LLM models, you can run on python scripts according to your needs, as below:
```python
python3 RLHF_train.py --dataset=sentiment-0 --batch_size=64 --num_epoch=5 --lr=5e-06 --rl_model_name=opt_large    --ref_model_name=opt_large --dropout=None --dropout_rate=0.0 --decoding=stochastic
```

Please refer to the shell files as of how to write the python scripts according to your needs 

# Reference
When you use reward dropout, please cite our paper !

```bibtex
@inproceedings{lee-lim-2024-towards,
    title = "Towards {P}areto-Efficient {RLHF}: Paying Attention to a Few High-Reward Samples with Reward Dropout",
    author = "Lee, Changhun  and
      Lim, Chiehyeon",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.489/",
    doi = "10.18653/v1/2024.findings-emnlp.489",
    pages = "8335--8349",
    abstract = "Recently, leveraging reinforcement learning (RL) to fine-tune language models (LMs), known as reinforcement learning from human feedback (RLHF), has become an important research topic. However, there is still a lack of theoretical understanding of how RLHF works, the conditions under which it succeeds or fails, and whether it guarantees optimization of both likelihood $\beta(\cdot)$ and reward $R(\cdot)$ objectives. To address these issues, we consider RLHF as a bi-objective problem that has the nature of a \textit{Pareto} optimization, present a Pareto improvement condition that is necessary to obtain Pareto-efficient policies, and propose a simple yet powerful method named \textit{reward dropout} that guarantees a Pareto improvement. To demonstrate the performance of reward dropout, two benchmark datasets commonly used in text style transfer tasks were utilized in our study: sentiment and topic datasets sourced from Yelp and AG{\_}News, respectively. Our experiments highlight that paying attention to a few samples with higher rewards leads to greater Pareto improvements regardless of model size. We also demonstrate that the effect of reward dropout is generalizable and most effective with non-pretrained target models, saving the effort of pretraining."
}