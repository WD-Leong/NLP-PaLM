# Import the libraries. #
import torch
import torch.nn.functional as F

# Multi-Head Attention Layer. #
class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self, d_model, n_heads, neg_infty=-1.0e9):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.neg_infty = neg_infty

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_depth = int(d_model / n_heads)
        
        self.wq = torch.nn.Linear(
            d_model, d_model, bias=False)
        self.wk = torch.nn.Linear(
            d_model, d_model, bias=False)
        self.wv = torch.nn.Linear(
            d_model, d_model, bias=False)
        self.wc = torch.nn.Linear(
            d_model, d_model, bias=False)
    
    def dot_attention(self, q, k, mask=None):
        lq = list(torch._shape_as_tensor(q))[2]
        lk = list(torch._shape_as_tensor(k))[2]
        
        # Head dimension. #
        dk = list(torch._shape_as_tensor(k))[-1]
        dk = torch.as_tensor(dk, dtype=torch.float32)
        
        # Multiplicative Attention. #
        matmul_qk = torch.matmul(
            q, torch.transpose(k, 2, 3))
        
        # Scale multiplicative attention mechanism. #
        attn_logits = matmul_qk * torch.rsqrt(dk)
        
        # Add the mask to the attention mechanism. #
        if mask is not None:
            attn_mask = (mask * self.neg_infty)
        else:
            attn_mask = torch.zeros(1, 1, lq, lk)
        
        if torch.cuda.is_available():
            attn_mask = attn_mask.to("cuda")
        
        attn_logits += attn_mask
        attn_weights = F.softmax(attn_logits, dim=3)
        return attn_weights

    def split_heads(self, x):
        input_dims = list(torch._shape_as_tensor(x))
        batch_size = input_dims[0]
        input_len  = input_dims[1]
        depth_size = int(input_dims[2] / self.n_heads)
        
        output_shape = tuple(
            [batch_size, input_len, self.n_heads, depth_size])
        output_heads = torch.reshape(x, output_shape)
        return torch.transpose(output_heads, 1, 2)

    def combine_heads(self, x):
        input_dims = list(torch._shape_as_tensor(x))
        batch_size = input_dims[0]
        input_len  = input_dims[2]
        num_heads  = input_dims[1]
        depth_size = input_dims[3]
        hidden_size = num_heads*depth_size
        
        output_shape  = tuple(
            [batch_size, input_len, hidden_size])
        output_tensor = torch.reshape(
            torch.transpose(x, 1, 2), output_shape)
        return output_tensor
    
    def forward(self, v, k, q, mask=None):
        q_input = self.split_heads(self.wq(q))
        k_input = self.split_heads(self.wk(k))
        v_input = self.split_heads(self.wv(v))
        
        attn_weights = self.dot_attention(
            q_input, k_input, mask=mask)
        attn_outputs = torch.matmul(
            attn_weights, v_input)
        attn_outputs = self.wc(
            self.combine_heads(attn_outputs))
        return attn_outputs, attn_weights

class FFWNetwork(torch.nn.Module):
    def __init__(self, d_ffwd, d_model):
        super(FFWNetwork, self).__init__()
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        
        self.ffwd_1 = torch.nn.Linear(d_model, d_ffwd)
        self.ffwd_2 = torch.nn.Linear(d_ffwd, d_model)
    
    def forward(self, x):
        # Use the square ReLU activation. #
        return self.ffwd_2(F.relu(self.ffwd_1(x)))

# Transformer Encoder Layer. #
class EncoderLayer(torch.nn.Module):
    def __init__(
        self, d_model, n_heads, d_ffwd, 
        rate1=0.1, rate2=0.1, neg_infty=-1.0e9):
        super(EncoderLayer, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.ffwd_self = FFWNetwork(d_ffwd, d_model)
        self.attn_self = MultiHeadAttention(
            d_model, n_heads, neg_infty=neg_infty)
        
        self.lnorm_1 = torch.nn.LayerNorm(d_model, eps=1.0e-6)
        self.lnorm_2 = torch.nn.LayerNorm(d_model, eps=1.0e-6)
        
        self.dropout_1 = torch.nn.Dropout(rate1)
        self.dropout_2 = torch.nn.Dropout(rate2)
    
    def forward(
        self, x_enc, x_pos, training=True):
        x_embed = x_enc + x_pos
        x_norm  = self.lnorm_1(x_embed)
        attn_self_tuple = self.attn_self(
            x_norm, x_norm, x_norm, mask=None)
        
        # Apply Normalisation followed by adding. #
        attn_self_output = torch.add(
            x_embed, attn_self_tuple[0])
        if training:
            attn_self_output = self.dropout_1(attn_self_output)
        
        ffwd_self_output = self.ffwd_self(
            self.lnorm_2(attn_self_output))
        ffwd_self_output = torch.add(
            attn_self_output, ffwd_self_output)
        if training:
            ffwd_self_output = self.dropout_2(ffwd_self_output)
        return ffwd_self_output

# Transformer Decoder Layer. #
class DecoderLayer(torch.nn.Module):
    def __init__(
        self, d_model, n_heads, d_ffwd, 
        rate1=0.1, rate2=0.1, neg_infty=-1.0e9):
        super(DecoderLayer, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.ffwd_self = FFWNetwork(d_ffwd, d_model)
        self.attn_self = MultiHeadAttention(
            d_model, n_heads, neg_infty=neg_infty)
        
        self.ffwd_enc_dec = FFWNetwork(d_ffwd, d_model)
        self.attn_enc_dec = MultiHeadAttention(
            d_model, n_heads, neg_infty=neg_infty)
        
        self.lnorm_1 = torch.nn.LayerNorm(d_model, eps=1.0e-6)
        self.lnorm_2 = torch.nn.LayerNorm(d_model, eps=1.0e-6)
        self.lnorm_3 = torch.nn.LayerNorm(d_model, eps=1.0e-6)
        self.lnorm_4 = torch.nn.LayerNorm(d_model, eps=1.0e-6)
        
        self.dropout_1 = torch.nn.Dropout(rate1)
        self.dropout_2 = torch.nn.Dropout(rate2)
    
    def forward(
        self, x_enc, x_dec, x_pos, training=True, mask=None):
        # Self-Attention Layer. #
        # Apply Normalisation followed by adding. #
        x_embed = x_dec + x_pos
        x_norm  = self.lnorm_1(x_embed)
        attn_self_tuple = self.attn_self(
            x_norm, x_norm, x_norm, mask=mask)
        
        attn_self_output = torch.add(
            x_embed, attn_self_tuple[0])
        if training:
            attn_self_output = self.dropout_1(attn_self_output)
        
        # Feed-Forward Layer. #
        ffwd_self_output = self.ffwd_self(
            self.lnorm_2(attn_self_output))
        ffwd_self_output = torch.add(
            attn_self_output, ffwd_self_output)
        if training:
            ffwd_self_output = self.dropout_2(ffwd_self_output)
        
        # Encoder-Decoder Attention Layer. #
        ffwd_self_norm_out = self.lnorm_3(ffwd_self_output)
        enc_dec_attn_tuple = self.attn_enc_dec(
            x_enc, x_enc, ffwd_self_norm_out, mask=None)
        
        enc_dec_attn_output = torch.add(
            ffwd_self_output, enc_dec_attn_tuple[0])
        if training:
            enc_dec_attn_output = self.dropout_1(enc_dec_attn_output)
        
        # Encoder-Decoder Feed-forward Layer. #
        enc_dec_ffwd_output = self.ffwd_enc_dec(
            self.lnorm_4(enc_dec_attn_output))
        enc_dec_ffwd_output = torch.add(
            enc_dec_attn_output, enc_dec_ffwd_output)
        if training:
            enc_dec_ffwd_output = self.dropout_2(enc_dec_ffwd_output)
        return enc_dec_ffwd_output

class Encoder(torch.nn.Module):
    def __init__(
        self, n_layers, d_model, n_heads, d_ffwd, 
        vocab_size, max_seq_length, rate1=0.1, rate2=0.1):
        super(Encoder, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.n_heads  = n_heads
        self.n_layers = n_layers
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.seq_len = max_seq_length
        self.d_rsqrt = torch.rsqrt(
            torch.as_tensor(d_model, dtype=torch.float32))
        self.vocab_size = vocab_size
        
        # Embedding Layers. #
        self.pos_embed = torch.nn.ModuleList()
        self.enc_embed = torch.nn.Embedding(vocab_size, d_model)
        
        # Encoder Layers. #
        self.enc_layers = torch.nn.ModuleList()
        for n_layer in range(n_layers):
            self.pos_embed.append(
                torch.nn.Embedding(max_seq_length, d_model))
            self.enc_layers.append(EncoderLayer(
                d_model, n_heads, d_ffwd, rate1, rate2))
        self.emb_dropout = torch.nn.Dropout(rate1)
        del n_layer
    
    def forward(self, x, training=True):
        seq_length  = list(torch._shape_as_tensor(x))[1]
        x_pos_index = torch.unsqueeze(torch.arange(
            0, seq_length, dtype=torch.long), 0)
        if torch.cuda.is_available():
            x_pos_index = x_pos_index.to("cuda")

        # Embed the tokens. #
        x_tok_embed = self.enc_embed(x)
        x_tok_embed = x_tok_embed * self.d_rsqrt
        if training:
            x_tok_embed = self.emb_dropout(x_tok_embed)
        
        layer_input = x_tok_embed
        for m in range(self.n_layers):
            x_pos_embed = self.pos_embed[m](x_pos_index)
            x_pos_embed = x_pos_embed * self.d_rsqrt
            if training:
                x_pos_embed = self.emb_dropout(x_pos_embed)
            
            layer_output = self.enc_layers[m](
                layer_input, x_pos_embed, training=training)
            layer_input  = layer_output
        return layer_output

class Decoder(torch.nn.Module):
    def __init__(
        self, n_layers, d_model, n_heads, d_ffwd, 
        vocab_size, max_seq_length, rate1=0.1, rate2=0.1):
        super(Decoder, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.n_heads  = n_heads
        self.n_layers = n_layers
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.seq_len = max_seq_length
        self.d_rsqrt = torch.rsqrt(
            torch.as_tensor(d_model, dtype=torch.float32))
        self.vocab_size = vocab_size

        # Embedding Layers. #
        self.pos_embed = torch.nn.ModuleList()
        self.dec_embed = torch.nn.Embedding(vocab_size, d_model)

        # Vocabulary index. #
        vocab_idx = torch.arange(
            0, vocab_size, dtype=torch.long)
        if torch.cuda.is_available():
            self.vocab_idx = vocab_idx.to("cuda")
        else:
            self.vocab_idx = vocab_idx
        del vocab_idx
        
        # Decoder Layers. #
        self.dec_layers = torch.nn.ModuleList()
        for n_layer in range(n_layers):
            self.pos_embed.append(
                torch.nn.Embedding(max_seq_length, d_model))
            self.dec_layers.append(DecoderLayer(
                d_model, n_heads, d_ffwd, rate1, rate2))
        self.emb_dropout = torch.nn.Dropout(rate1)
        del n_layer
    
    def forward(self, x_enc, x_dec, training=True):
        seq_length = list(torch._shape_as_tensor(x_dec))[1]
        input_zero = torch.zeros(seq_length, seq_length)
        input_mask = torch.triu(1.0 + input_zero, diagonal=1)
        input_mask = torch.unsqueeze(
            torch.unsqueeze(input_mask, 0), 0)
        
        x_pos_index = torch.unsqueeze(torch.arange(
            0, seq_length, dtype=torch.long), 0)
        if torch.cuda.is_available():
            x_pos_index = x_pos_index.to("cuda")

        # Embed the tokens. #
        x_tok_embed = self.dec_embed(x_dec)
        x_tok_embed = x_tok_embed * self.d_rsqrt
        if training:
            x_tok_embed = self.emb_dropout(x_tok_embed)
        
        layer_input = x_tok_embed
        for m in range(self.n_layers):
            x_pos_embed = self.pos_embed[m](x_pos_index)
            x_pos_embed = x_pos_embed * self.d_rsqrt
            if training:
                x_pos_embed = self.emb_dropout(x_pos_embed)
            
            layer_output = self.dec_layers[m](
                x_enc, layer_input, x_pos_embed, 
                training=training, mask=input_mask)
            layer_input  = layer_output
        
        # Extract the logits. #
        w_embedding = self.dec_embed(self.vocab_idx)
        dec_logits  = torch.matmul(
            layer_output, torch.transpose(w_embedding, 0, 1))
        return layer_output, dec_logits

class Transformer(torch.nn.Module):
    def __init__(
        self, n_layers, n_heads, d_model, d_ffwd, 
        enc_vocab_size, dec_vocab_size, enc_seq_length, 
        dec_seq_length, rate1=0.1, rate2=0.1):
        super(Transformer, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.n_heads  = n_heads
        self.n_layers = n_layers
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.enc_seq_length = enc_seq_length
        self.dec_seq_length = dec_seq_length
        self.enc_vocab_size = enc_vocab_size
        self.dec_vocab_size = dec_vocab_size
        
        self.ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        
        # Output projection. #
        self.encoder = Encoder(
            n_layers, d_model, n_heads, d_ffwd, 
            enc_vocab_size, enc_seq_length, rate1=rate1, rate2=rate2)
        self.decoder = Decoder(
            n_layers, d_model, n_heads, d_ffwd, 
            dec_vocab_size, dec_seq_length, rate1=rate1, rate2=rate2)
    
    def forward(self, x_encode, x_decode, training=True):
        enc_outputs = self.encoder(
            x_encode, training=training)
        dec_logits = self.decoder(
            enc_outputs, x_decode, training=training)
        return dec_logits
    
    def compute_ce_loss(self, x_encode, x_decode, x_labels):
        enc_ouputs = self.encoder(x_encode, training=True)
        dec_logits = self.decoder(
            enc_ouputs, x_decode, training=True)[1]
        
        raw_ce_loss = []
        for step in range(self.dec_seq_length):
            raw_ce_loss.append(torch.unsqueeze(self.ce_loss_fn(
                dec_logits[:, step, :], x_labels[:, step]), 1))
        
        seq_ce_loss = torch.cat(raw_ce_loss, 1)
        tot_ce_loss = torch.sum(torch.sum(seq_ce_loss, 1))
        return tot_ce_loss
    
    def infer(self, x_enc, x_sos, k=1):
        infer_ids = [torch.unsqueeze(x_sos[:, 0], 1)]
        
        with torch.no_grad():
            # Encoder. #
            enc_outputs = self.encoder(x_enc, training=False)

            for step in range(self.dec_seq_length):
                tmp_inputs = torch.cat(infer_ids, dim=1)
            
                tmp_logits = self.decoder(
                    enc_outputs, tmp_inputs, training=False)[1]
                
                if k == 1:
                    tmp_argmax =  torch.argmax(
                        tmp_logits[:, -1, :], dim=1)
                    infer_ids.append(torch.unsqueeze(tmp_argmax, 1))
                else:
                    tmp_logit = tmp_logits[:, -1, :]
                    tmp_prob  = F.softmax(tmp_logit, dim=1)

                    tmp_top_k  = torch.topk(tmp_prob, k=k)
                    tmp_sample = torch.multinomial(
                        tmp_top_k.values, 1)
                    tmp_index  = torch.gather(
                        tmp_top_k.indices, 1, tmp_sample)
                    infer_ids.append(tmp_index)
        return torch.cat(infer_ids, dim=1)
