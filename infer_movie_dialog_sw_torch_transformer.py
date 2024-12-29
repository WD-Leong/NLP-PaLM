# Import the libraries. #
import time
import numpy as np
import pandas as pd
import pickle as pkl
import byte_pair_encoding as bpe

import torch
import torch_transformer as transformer_module

# Model Parameters. #
i_seq_len  = 15
o_seq_len  = 15
num_heads  = 4
num_layers = 3

p_keep = 0.9
p_drop = 1.0 - p_keep

hidden_size = 256
ffwd_size   = 4*hidden_size

model_ckpt_dir  = "../PyTorch_Models/dialog_sw_torch_transformer"
train_loss_file = "train_loss_dialog_sw_torch_transformer.csv"

# Load the data. #
tmp_pkl_file = "../Data/movie_dialogs/"
tmp_pkl_file += "movie_dialogues_sw.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    data_tuple = pkl.load(tmp_load_file)
    subword_vocab = pkl.load(tmp_load_file)
    idx_2_subword = pkl.load(tmp_load_file)
    subword_2_idx = pkl.load(tmp_load_file)

vocab_size = len(subword_vocab)
print("Subword Vocabulary Size:", str(vocab_size) + ".")

SOS_token = subword_2_idx["<SOS>"]
EOS_token = subword_2_idx["<EOS>"]
PAD_token = subword_2_idx["<PAD>"]
UNK_token = subword_2_idx["<UNK>"]

# Set the number of threads to use. #
torch.set_num_threads(1)

# Build the Transformer network. #
print("Building the Transformer Model.")
start_time = time.time()

transformer_model = transformer_module.Transformer(
    num_layers, num_heads, hidden_size, ffwd_size, vocab_size, 
    vocab_size, i_seq_len+2, o_seq_len+1, rate1=0.0, rate2=p_drop)
if torch.cuda.is_available():
    transformer_model.to("cuda")

transformer_optim = torch.optim.Adam(
    transformer_model.parameters(), 
    weight_decay=1.0e-4, eps=1.0e-7)

elapsed_time = (time.time()-start_time) / 60
print("Transformer Model Built (" + str(elapsed_time), "mins).")

ckpt = torch.load(model_ckpt_dir)
n_iter = ckpt["step"]

transformer_model.load_state_dict(ckpt['model_state_dict'])
transformer_optim.load_state_dict(ckpt['optim_state_dict'])

train_loss_df = pd.read_csv(train_loss_file)
train_loss_list = [tuple(
    train_loss_df.iloc[x].values) \
    for x in range(len(train_loss_df))]

# GPT model inference. #
print("-" * 50)
print("Transformer Model Inference", 
      "(" + str(n_iter) + " iterations).")
print("-" * 50)

# Set the model to eval. #
transformer_model.eval()

# Start inferring. #
while True:
    tmp_phrase = input("Enter input: ")
    tmp_phrase = tmp_phrase.lower().strip()

    if tmp_phrase == "":
        break
    else:
        tmp_i_idx = [SOS_token]
        tmp_i_idx += bpe.bp_encode(
            tmp_phrase, subword_vocab, subword_2_idx)
        tmp_i_idx += [EOS_token]
        n_tokens  = len(tmp_i_idx)
        n_padding = i_seq_len + 2 - n_tokens
        
        if n_padding <= 0:
            pad_sequence = []
        else:
            pad_sequence = [PAD_token] * n_padding
        
        tmp_test_in  = np.array(
            tmp_i_idx + pad_sequence).reshape((1, -1))
        tmp_test_sos = np.array([SOS_token]).reshape((1, -1))

        infer_in  = torch.tensor(
            tmp_test_in, dtype=torch.long)
        infer_sos = torch.tensor(
            tmp_test_sos, dtype=torch.long)
        if torch.cuda.is_available():
            infer_in  = infer_in.to("cuda")
            infer_sos = infer_sos.to("cuda")
        
        tmp_infer = transformer_model.infer(
            infer_in, infer_sos, k=1)
        if torch.cuda.is_available():
            tmp_infer = tmp_infer.detach().cpu()
        
        gen_phrase = bpe.bp_decode(
            tmp_infer[0].numpy(), idx_2_subword)
        gen_phrase = " ".join(
            gen_phrase).replace("<", "").replace(">", "")
        del n_tokens

        print("")
        print("Input Phrase:")
        print(tmp_phrase)
        print("Generated Reply:")
        print(gen_phrase)
        print("-" * 50)
