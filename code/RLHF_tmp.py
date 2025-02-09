# %%
'''
라이브러리 로드
'''
import os
from pathlib import Path
import argparse
import json
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import time
import glob
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

import einops
from utils import createFolder
from tensorflow.keras.utils import Progbar

# %%
old_lm_head_decoder = rl_model.get_output_embeddings().kernel
old_embedding_dim = tf.shape(old_lm_head_decoder)[0]
decoder_mask, current_decoder = init_copy_embeddings(old_lm_head_decoder, len(rl_model_tokenizer))
new_lm_head_decoder = old_lm_head_decoder
new_lm_head_decoder = new_lm_head_decoder.add_weight(
    shape=(old_embedding_dim, new_num_tokens),
    initializer="zeros",
    trainable=True,
    name=old_lm_head_decoder.name.split(":")[0],
)
init_decoder = tf.where(decoder_mask, current_decoder, new_lm_head_decoder.value())

new_lm_head_decoder.assign(init_decoder)



# %%
from transformers import TFSharedEmbeddings
def custom_resize_token_embeddings(model, new_num_tokens):

    def init_copy_embeddings(old_embeddings, new_num_tokens):
        old_num_tokens, old_embedding_dim = tf.shape(old_embeddings)[1], tf.shape(old_embeddings)[0]
        size_diff = new_num_tokens - old_num_tokens

        # initialize new embeddings
        # Copy token embeddings from the previous ones
        if tf.math.greater(size_diff, 0):
            # if the new size is greater than the old one, we extend the current embeddings with a padding until getting new size
            # and we create a mask to properly identify the padded values and be replaced by the values of the newly created
            # embeddings
            current_weights = tf.pad(
                old_embeddings.value(), tf.convert_to_tensor([[0, size_diff], [0, 0]]), constant_values=-1
            )
            num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
            mask = tf.fill(tf.convert_to_tensor([num_tokens_to_copy, 1]), True)
            mask = tf.pad(mask, tf.convert_to_tensor([[0, size_diff], [0, 0]]), constant_values=False)
        else:
            # if the new size if lower than the old one, we take the current embeddings until the new size
            current_weights = tf.slice(
                old_embeddings.value(),
                tf.convert_to_tensor([0, 0]),
                tf.convert_to_tensor([new_num_tokens, old_embedding_dim]),
            )
            mask = tf.fill(tf.convert_to_tensor([new_num_tokens, 1]), True)

        return mask, current_weights

    def get_resized_lm_head_decoder(old_lm_head_decoder, new_num_tokens):

        if old_lm_head_decoder is not None:
            old_embedding_dim = tf.shape(old_lm_head_decoder)[0]
            decoder_mask, current_decoder = init_copy_embeddings(old_lm_head_decoder, new_num_tokens)
            new_lm_head_decoder = new_lm_head_decoder.add_weight(
                shape=(old_embedding_dim, new_num_tokens),
                initializer="zeros",
                trainable=True,
                name=old_lm_head_decoder.name.split(":")[0],
            )
            init_decoder = tf.where(decoder_mask, current_decoder, new_lm_head_decoder.value())

            new_lm_head_decoder.assign(init_decoder)

        return new_lm_head_decoder


    def get_embedding_weight(embedding, input_ids=None):
        if isinstance(embedding, TFSharedEmbeddings):
            if input_ids is None:
                input_ids = tf.range(embedding.vocab_size)
            return embedding(input_ids)

        elif isinstance(embedding, tf.keras.layers.Embedding):
            return embedding.embeddings

        elif isinstance(embedding, tf.keras.layers.Dense):
            return embedding.kernel

        elif isinstance(embedding, tf.Variable):
            return embedding

        else:
            raise ValueError(f"Unsupported embedding type: {type(embedding)}")

    def resize_embedding_layer(embedding, new_num_tokens, input_ids=None):
        if isinstance(embedding, TFSharedEmbeddings):
            old_embedding = get_embedding_weight(embedding, input_ids)
            old_num_tokens, old_embedding_dim = tf.shape(old_embedding)[0], tf.shape(old_embedding)[1]

            size_diff = new_num_tokens - old_num_tokens

            if size_diff > 0:
                # Create new embeddings with zero initialization for the additional tokens
                new_embeddings = tf.concat([old_embedding, tf.zeros([size_diff, old_embedding_dim])], axis=0)
            else:
                # Trim the embeddings if reducing the number of tokens
                new_embeddings = old_embedding[:new_num_tokens]

            return new_embeddings

        else:
            old_embedding = get_embedding_weight(embedding, input_ids)
            old_num_tokens, old_embedding_dim = tf.shape(old_embedding)[1], tf.shape(old_embedding)[0]

            size_diff = new_num_tokens - old_num_tokens

            if size_diff > 0:
                # Create new embeddings with zero initialization for the additional tokens
                new_embeddings = tf.concat([old_embedding, tf.zeros([old_embedding_dim, size_diff])], axis=1)   # new_embeddings = (1024, 256011)
            else:
                # Trim the embeddings if reducing the number of tokens
                new_embeddings = old_embedding[:new_num_tokens]

            return new_embeddings

    dummy_input_ids = tf.range(model.config.vocab_size)

    # Resize input embeddings
    old_input_embeddings = model.get_input_embeddings()
    new_input_embeddings = resize_embedding_layer(old_input_embeddings, new_num_tokens, dummy_input_ids)

    if isinstance(old_input_embeddings, TFSharedEmbeddings):
        new_shared_embedding = TFSharedEmbeddings(
            new_num_tokens,
            old_input_embeddings(dummy_input_ids).shape[-1]
        )
        new_shared_embedding.build((new_num_tokens,))
        new_shared_embedding.set_weights([new_input_embeddings])
        model.set_input_embeddings(new_shared_embedding)
    else:
        old_input_embeddings.set_weights([new_input_embeddings])

    # Resize output embeddings if they exist
    if model.get_output_embeddings() is not None:
        old_output_embeddings = model.get_output_embeddings()

        if isinstance(old_output_embeddings, tf.keras.layers.Dense):
            old_output_weights = old_output_embeddings.kernel
            new_output_weights = resize_embedding_layer(old_output_weights, new_num_tokens, dummy_input_ids)    # new_embeddings = (1024, 256011)
            old_output_embeddings.kernel.assign(new_output_weights)

            # new_output_embeddings = resize_embedding_layer(old_output_embeddings, new_num_tokens, dummy_input_ids)    # new_embeddings = (1024, 256011)
            # new_shared_embedding = TFSharedEmbeddings(
            #     old_output_embeddings.kernel.shape[0],
            #     new_num_tokens
            # )
            # new_shared_embedding.build((new_num_tokens,))
            # new_shared_embedding.set_weights([new_output_embeddings])
            # model.set_output_embeddings(new_shared_embedding)


            # if old_output_embeddings.bias is not None:
            #     old_output_bias = old_output_embeddings.bias
            #     size_diff = new_num_tokens - tf.shape(old_output_bias)[0]
            #     new_output_bias = tf.concat([old_output_bias, tf.zeros([size_diff])], axis=0) if size_diff > 0 else old_output_bias[:new_num_tokens]
            #     old_output_embeddings.bias.assign(new_output_bias)
        else:
            new_output_embeddings = resize_embedding_layer(old_output_embeddings, new_num_tokens, dummy_input_ids)

            if isinstance(old_output_embeddings, TFSharedEmbeddings):
                new_shared_embedding = TFSharedEmbeddings(
                    new_num_tokens,
                    old_output_embeddings(dummy_input_ids).shape[-1]
                )
                new_shared_embedding.build((new_num_tokens,))
                new_shared_embedding.set_weights([new_output_embeddings])
                model.set_output_embeddings(new_shared_embedding)
            else:
                old_output_embeddings.set_weights(new_output_embeddings)

    # Resize biases if they exist
    if model.get_bias() is not None:
        old_lm_head_bias = model.get_bias()
        new_lm_head_bias = model._v2_get_resized_lm_head_bias(old_lm_head_bias, new_num_tokens)
        model.set_bias(new_lm_head_bias)

    return model.get_input_embeddings()