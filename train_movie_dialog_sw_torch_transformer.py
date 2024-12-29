# Import the libraries. #
import time
import numpy as np
import pandas as pd
import pickle as pkl
import byte_pair_encoding as bpe

import torch
import torch_transformer as transformer_module

# Define the weight update step for multiple sub-batches. #
def sub_batch_train_step(
    model, optimizer, sub_batch_sz, 
    x_encode, x_output, grad_clip=1.0):
    batch_size = x_encode.shape[0]
    if batch_size <= sub_batch_sz:
        sub_batch = 1
    elif batch_size % sub_batch_sz == 0:
        sub_batch = int(batch_size / sub_batch_sz)
    else:
        sub_batch = int(batch_size / sub_batch_sz) + 1
    avg_batch_loss = 0.0
    
    optimizer.zero_grad()
    for n_sub in range(sub_batch):
        id_st = n_sub*sub_batch_sz
        if n_sub != (sub_batch-1):
            id_en = (n_sub+1)*sub_batch_sz
        else:
            id_en = batch_size
        
        tmp_encode = torch.tensor(
            x_encode[id_st:id_en, :], dtype=torch.long)
        tmp_decode = torch.tensor(
            x_output[id_st:id_en, :-1], dtype=torch.long)
        tmp_labels = torch.tensor(
            x_output[id_st:id_en, 1:], dtype=torch.long)

        if torch.cuda.is_available():
            tmp_encode = tmp_encode.to("cuda")
            tmp_decode = tmp_decode.to("cuda")
            tmp_labels = tmp_labels.to("cuda")
        
        # Calculate the cross entropy loss. #
        tmp_ce_loss = model.compute_ce_loss(
            tmp_encode, tmp_decode, tmp_labels)
        
        # Accumulate the gradients. #
        sub_batch_loss = tmp_ce_loss / batch_size
        avg_batch_loss += sub_batch_loss.item()
        sub_batch_loss.backward()
    
    # Update using the optimizer. #
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), grad_clip)
    optimizer.step()
    return avg_batch_loss

# Model Parameters. #
batch_size = 256
sub_batch  = 256
i_seq_len  = 15
o_seq_len  = 15
num_heads  = 4
num_layers = 3

gradient_clip = 1.00
maximum_iter  = 8000
restore_flag  = True
save_step     = 250
warmup_steps  = 5000
display_step  = 50

p_keep = 0.9
p_drop = 1.0 - p_keep

hidden_size = 256
ffwd_size   = 4*hidden_size
cooling_step = 250

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

# Filter the dataset. #
filtered_data = []
for tmp_data in data_tuple:
    i_len = len(tmp_data[0])
    o_len = len(tmp_data[1])
    if i_len <= i_seq_len and o_len <= o_seq_len:
        filtered_data.append(tmp_data)
del tmp_data, data_tuple

data_tuple = filtered_data
vocab_size = len(subword_vocab)
print("Subword Vocabulary Size:", str(vocab_size) + ".")
del filtered_data

num_data  = len(data_tuple)
SOS_token = subword_2_idx["<SOS>"]
EOS_token = subword_2_idx["<EOS>"]
PAD_token = subword_2_idx["<PAD>"]
UNK_token = subword_2_idx["<UNK>"]
print("Total of", num_data, "rows loaded.")

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

transformer_optim = torch.optim.AdamW(
    transformer_model.parameters(), weight_decay=1.0e-4)

elapsed_time = (time.time()-start_time) / 60
print("Transformer Model Built (" + str(elapsed_time), "mins).")

if restore_flag:
    ckpt = torch.load(model_ckpt_dir)
    n_iter = ckpt["step"]
    
    transformer_model.load_state_dict(ckpt['model_state_dict'])
    transformer_optim.load_state_dict(ckpt['optim_state_dict'])
    
    train_loss_df = pd.read_csv(train_loss_file)
    train_loss_list = [tuple(
        train_loss_df.iloc[x].values) \
        for x in range(len(train_loss_df))]
else:
    print("Training a new model.")
    n_iter = 0
    train_loss_list = []

# Train the Transformer model. #
tmp_in_seq  = np.zeros(
    [batch_size, i_seq_len+2], dtype=np.int32)
tmp_out_seq = np.zeros(
    [batch_size, o_seq_len+2], dtype=np.int32)

# Warmup learning schedule. #
step_min = float(max(n_iter, warmup_steps))**(-0.5)
learning_rate = float(hidden_size)**(-0.5) * step_min

print("-" * 50)
print("Training the Transformer Network", 
      "(" + str(n_iter) + " iterations).")
print(str(num_data), "training samples.")
print("-" * 50)

# Update the neural network's weights. #
tot_loss = 0.0
start_tm = time.time()
while n_iter < maximum_iter:
    step_min = float(max(n_iter, warmup_steps))**(-0.5)
    learning_rate = float(hidden_size)**(-0.5) * step_min
    
    for tmp_opt_group in transformer_optim.param_groups:
        tmp_opt_group["lr"] = learning_rate
    
    # Select a sample from the data. #
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    
    tmp_in_seq[:, :]  = PAD_token
    tmp_out_seq[:, :] = PAD_token
    for n_index in range(batch_size):
        tmp_index = batch_sample[n_index]
        tmp_i_idx = [SOS_token]
        tmp_i_idx += data_tuple[tmp_index][0] + [EOS_token]
        tmp_o_idx = [SOS_token]
        tmp_o_idx += data_tuple[tmp_index][1] + [EOS_token]
        
        n_input  = len(tmp_i_idx)
        n_output = len(tmp_o_idx)
        tmp_in_seq[n_index, :n_input] = tmp_i_idx
        tmp_out_seq[n_index, :n_output] = tmp_o_idx
    
    tmp_loss = sub_batch_train_step(
        transformer_model, transformer_optim, 
        sub_batch, tmp_in_seq, tmp_out_seq, grad_clip=1.0)
    
    n_iter += 1
    tot_loss += tmp_loss
    if n_iter % display_step == 0:
        end_tm = time.time()
        
        avg_loss = tot_loss / display_step
        tot_loss = 0.0
        elapsed_tm = (end_tm - start_tm) / 60
        
        sample_id = np.random.choice(num_data)
        tmp_data  = data_tuple[sample_id]
        tmp_i_idx = [SOS_token] + tmp_data[0] + [EOS_token]
        tmp_o_idx = [SOS_token] + tmp_data[1] + [EOS_token]
        tmp_i_tok = bpe.bp_decode(tmp_i_idx, idx_2_subword)
        tmp_o_tok = bpe.bp_decode(tmp_o_idx, idx_2_subword)
        n_padding = i_seq_len - len(tmp_data[0])
        
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
        
        # Set the model to eval. #
        transformer_model.eval()
        tmp_infer = transformer_model.infer(
            infer_in, infer_sos, k=1)
        if torch.cuda.is_available():
            tmp_infer = tmp_infer.detach().cpu()
        del tmp_data

        tmp_in_phrase  = " ".join(
            tmp_i_tok).replace("<", "").replace(">", "")
        tmp_out_phrase  = " ".join(
            tmp_o_tok).replace("<", "").replace(">", "")
        
        gen_tokens = tmp_infer[0].numpy()
        gen_phrase = bpe.bp_decode(
            gen_tokens, idx_2_subword)
        gen_phrase = " ".join(
            gen_phrase).replace("<", "").replace(">", "")
        
        print("Iteration", str(n_iter) + ".")
        print("Elapsed Time:", str(elapsed_tm), "mins.")
        print("Gradient Clip:", str(gradient_clip) + ".")
        print("Learning Rate:", str(learning_rate) + ".")
        print("Average Loss:", str(avg_loss) + ".")
        
        print("")
        print("Input Phrase:")
        print(tmp_in_phrase)
        print("Generated Reply:")
        print(gen_phrase)
        print("Actual Response:")
        print(tmp_out_phrase)
        
        # Set the model back to training. #
        transformer_model.train()
        
        # Append the training loss. #
        train_loss_list.append((n_iter, avg_loss))
        
        # Reset the timer. #
        start_tm = time.time()
        print("-" * 50)
    
    # Save the model. #
    if n_iter % save_step == 0:
        # Save the model. #
        torch.save({
            'step': n_iter,
            'model_state_dict': transformer_model.state_dict(),
            'optim_state_dict': transformer_optim.state_dict()
            }, model_ckpt_dir)
        print("Saved model to:", model_ckpt_dir)
        
        tmp_df_losses = pd.DataFrame(
            train_loss_list, columns=["n_iter", "xent_loss"])
        tmp_df_losses.to_csv(train_loss_file, index=False)
        del tmp_df_losses
    
    # Cool the GPU. #
    if n_iter % cooling_step == 0:
        print("Cooling GPU for 2.5 minutes.")
        time.sleep(150)
        print("Resume Training.")
        print("-" * 50)