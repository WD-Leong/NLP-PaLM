import time
import numpy as np
import pandas as pd
import pickle as pkl
import byte_pair_encoding as bpe

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_transformer_palm_keras as tf_transformer

# Model Parameters. #
seq_encode = 15
seq_decode = 16

num_layers  = 3
num_heads   = 4
prob_keep   = 0.9
hidden_size = 256
ffwd_size   = 4*hidden_size

model_ckpt_dir  = "../TF_Models/dialogue_sw_transformer_palm"
train_loss_file = "train_loss_dialogue_sw_transformer_palm.csv"

tmp_pkl_file = "../Data/movie_dialogs/movie_dialogues_sw.pkl"
with open(tmp_pkl_file, "rb") as tmp_file_load:
    data_tuple = pkl.load(tmp_file_load)
    subword_vocab = pkl.load(tmp_file_load)
    idx_2_subword = pkl.load(tmp_file_load)
    subword_2_idx = pkl.load(tmp_file_load)

# Filter the dataset. #
filtered_data = []
for tmp_data in data_tuple:
    if len(tmp_data[0]) <= seq_encode \
        and len(tmp_data[1]) <= (seq_decode-1):
        filtered_data.append(tmp_data)
del tmp_data, data_tuple

data_tuple = filtered_data
vocab_size = len(subword_vocab)
print("Vocabulary Size:", str(vocab_size))
del filtered_data

num_data  = len(data_tuple)
SOS_token = subword_2_idx["<SOS>"]
EOS_token = subword_2_idx["<EOS>"]
PAD_token = subword_2_idx["<PAD>"]
UNK_token = subword_2_idx["<UNK>"]

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

print("Building the Transformer Model.")
start_time = time.time()

seq2seq_model = tf_transformer.Transformer(
    num_layers, num_heads, hidden_size, 
    ffwd_size, vocab_size, vocab_size, seq_encode, 
    seq_decode, rate1=0.0, rate2=1.0-prob_keep)
seq2seq_optim = tfa.optimizers.AdamW(
    weight_decay=1.0e-4)

elapsed_time = (time.time() - start_time) / 60
print("Transformer Model built", 
      "(" + str(elapsed_time) + " mins).")

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    seq2seq_model=seq2seq_model, 
    seq2seq_optim=seq2seq_optim)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Model restored from {}".format(
        manager.latest_checkpoint))
else:
    print("Error: No latest checkpoint found.")

train_loss_df = pd.read_csv(train_loss_file)
train_loss_list = [tuple(
    train_loss_df.iloc[x].values) \
        for x in range(len(train_loss_df))]

# Placeholders to store the batch data. #
tmp_test_in = np.zeros(
    [1, seq_encode], dtype=np.int32)

n_iter = ckpt.step.numpy().astype(np.int32)
print("-" * 50)
print("Inference of Transformer Keras Network", 
      "(" + str(n_iter), "iterations).")
print("Total of", str(num_data), "training samples.")

while True:
    tmp_in_phrase = input("Enter input phrase: ")
    tmp_in_phrase = tmp_in_phrase.strip().lower()
    
    if tmp_in_phrase == "":
        break
    else:
        tmp_i_idx = bpe.bp_encode(
            tmp_in_phrase, subword_vocab, subword_2_idx)
        
        n_input = len(tmp_i_idx)
        if n_input >= seq_encode:
            print("Input sequence too long.")
            continue
        else:
            tmp_i_tok = bpe.bp_decode(tmp_i_idx, idx_2_subword)
            n_sw_toks = len(tmp_i_idx)

            tmp_test_in[:, :] = PAD_token
            tmp_test_in[0, :n_sw_toks] = tmp_i_idx
            
            gen_ids = seq2seq_model.infer(
                tmp_test_in, SOS_token)
            gen_phrase = bpe.bp_decode(
                gen_ids.numpy()[0], idx_2_subword)
        
        print("Input Phrase:")
        print(" ".join(tmp_i_tok).replace("<", "").replace(">", ""))
        print("Generated Phrase:")
        print(" ".join(gen_phrase).replace("<", "").replace(">", ""))
        print("-" * 50)
