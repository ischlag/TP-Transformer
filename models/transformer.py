# The Attention-Is-All-You-Need Transformer
#
# Based on the official implementation: https://github.com/tensorflow/models/
# blob/master/official/transformer/model/transformer.py
#
# Input:
#   tok_emb = embedding(token)  # init N(0,1)
#   pos_emb = sine_cosine_encoding(position)  # section 3.4
#   enc_inputs = dropout(tok_emb + pos_emb)
#
# EncoderLayer:
#   sublayer 1:
#     z = layernorm(x)
#     z = MHA(z)
#     z = dropout(z)
#     x += z
#   sublayer 2:
#     z = layernorm(x)
#     z = densefilter(z)
#     z = dropout(z)
#     x += z
#   layernorm(x)
#
# Encoder:
#   Input
#   6x EncoderLayer
#
# DecoderLayer:
#   sublayer 1:
#     z = layernorm(x)
#     z = MHA(z)
#     z = dropout(z)
#     x += z
#   sublayer 2:
#     z = layernorm(x)
#     z = MHA_over_encoder_outputs(z)
#     z = dropout(z)
#     x += z
#   sublayer 3:
#     z = layernorm(x)
#     z = densefilter(z)
#     z = dropout(z)
#     x += z
#   layernorm(x)
#
# Decoder:
#   Input (use encoder embedding matrix)
#   6x DecoderLayer
#
# Model:
#   dec_output = Decoder(Encoder(Input))
#   logits = linear(dec_output)  # use embedding transposed as weights
#   loss = cross_entropy(logits)
#
# Errata:
# 1.) This implementation uses throughout affine transformations
#     instead of linear ones
#

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_transformer(params, pad_idx):

  embedding = TokenEmbedding(d_vocab=params.input_dim,
                             d_h=params.hidden,
                             d_p=params.hidden,
                             dropout=params.dropout,
                             max_length=200)

  encoder = Encoder(hid_dim=params.hidden,
                    n_layers=params.n_layers,
                    n_heads=params.n_heads,
                    pf_dim=params.filter,
                    encoder_layer=EncoderLayer,
                    self_attention=SelfAttention,
                    positionwise_feedforward=PositionwiseFeedforward,
                    dropout=params.dropout)

  decoder = Decoder(hid_dim=params.hidden,
                    n_layers=params.n_layers,
                    n_heads=params.n_heads,
                    pf_dim=params.filter,
                    decoder_layer=DecoderLayer,
                    self_attention=SelfAttention,
                    positionwise_feedforward=PositionwiseFeedforward,
                    dropout=params.dropout)

  model = Seq2Seq(embedding=embedding,
                  encoder=encoder,
                  decoder=decoder,
                  pad_idx=pad_idx)

  return model


class TokenEmbedding(nn.Module):
  def __init__(self, d_vocab, d_h, d_p, dropout, max_length):
    super(TokenEmbedding, self).__init__()
    self.dropout = nn.Dropout(dropout)

    # token encodings
    self.d_h = d_h
    self.tok_embedding = nn.Embedding(d_vocab, d_h)
    self.scale = torch.sqrt(torch.FloatTensor([d_h]))

    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_length, d_p)
    position = torch.arange(0., max_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0., d_p, 2) *
                         -(math.log(10000.0) / d_p))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)
    # pe = [1, seq_len, d_p]

    self.reset_parameters()  # init tok_embedding to N(0,1/sqrt(d_h))

  def forward(self, src):
    # src = [batch_size, src_seq_len]

    # scale up embedding to be N(0,1)
    tok_emb = self.tok_embedding(src) * self.scale.to(src.device)
    pos_emb = torch.autograd.Variable(self.pe[:, :src.size(1)],
                                      requires_grad=False)
    x = tok_emb + pos_emb
    x = self.dropout(x)

    # src = [batch_size, src_seq_len, d_h]
    return x

  def transpose_forward(self, trg):
    # trg = [batch_size, trg_seq_len, d_h]
    logits = torch.einsum('btd,vd->btv',trg,self.tok_embedding.weight)
    # logits = torch.matmul(trg, torch.transpose(self.tok_embedding.weight, 0, 1))
    # logits = [batch_size, trg_seq_len, d_vocab]
    return logits

  def reset_parameters(self):
    nn.init.normal_(self.tok_embedding.weight,
                    mean=0,
                    std=1./math.sqrt(self.d_h))


class Encoder(nn.Module):
  def __init__(self, hid_dim, n_layers, n_heads, pf_dim,
               encoder_layer, self_attention, positionwise_feedforward, dropout):
    super().__init__()

    self.layers = nn.ModuleList([encoder_layer(hid_dim, n_heads, pf_dim,
                                               self_attention,
                                               positionwise_feedforward,
                                               dropout)
                                 for _ in range(n_layers)])


  def forward(self, src, src_mask):
    # src = [batch_size, src_seq_len]
    # src_mask = [batch_size, src_seq_len]
    for layer in self.layers:
      src = layer(src, src_mask)

    return src


class EncoderLayer(nn.Module):
  def __init__(self, hid_dim, n_heads, pf_dim, self_attention,
               positionwise_feedforward, dropout):
    super().__init__()

    self.layernorm1 = nn.LayerNorm(hid_dim)
    self.layernorm2 = nn.LayerNorm(hid_dim)
    self.layernorm3 = nn.LayerNorm(hid_dim)
    self.MHA = self_attention(hid_dim, n_heads, dropout)
    self.densefilter = positionwise_feedforward(hid_dim, pf_dim, dropout)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)


  def forward(self, src, src_mask):
    # src = [batch_size, src_seq_size, hid_dim]
    # src_mask = [batch_size, src_seq_size]

    # sublayer 1
    z = self.layernorm1(src)
    z, attn = self.MHA(z, z, z, src_mask)
    z = self.dropout1(z)
    src = src + z

    # sublayer 2
    z = self.layernorm2(src)
    z = self.densefilter(z)
    z = self.dropout2(z)
    src = src + z

    return self.layernorm3(src)


class SelfAttention(nn.Module):
  def __init__(self, hid_dim, n_heads, dropout):
    super().__init__()

    self.hid_dim = hid_dim
    self.n_heads = n_heads

    assert hid_dim % n_heads == 0

    self.w_q = nn.Linear(hid_dim, hid_dim)
    self.w_k = nn.Linear(hid_dim, hid_dim)
    self.w_v = nn.Linear(hid_dim, hid_dim)

    self.linear = nn.Linear(hid_dim, hid_dim)
    self.dropout = nn.Dropout(dropout)
    self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    self.reset_parameters()

  def forward(self, query, key, value, mask=None):
    # query = key = value = [batch_size, seq_len, hid_dim]
    # src_mask = [batch_size, 1, 1, pad_seq]
    # trg_mask = [batch_size, 1, pad_seq, past_seq]

    bsz = query.shape[0]

    Q = self.w_q(query)
    K = self.w_k(key)
    V = self.w_v(value)
    # Q, K, V = [batch_size, seq_len, hid_dim]

    Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads)\
         .permute(0,2,1,3)
    K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads)\
         .permute(0,2,1,3)
    V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads)\
         .permute(0,2,1,3)
    # Q, K, V = [batch_size, n_heads, seq_size, hid_dim // n heads]

    energy = torch.einsum('bhid,bhjd->bhij',Q,K) / self.scale.to(key.device)
    # energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(key.device)

    # energy   = [batch_size, n_heads, query_pos     , key_pos]
    # src_mask = [batch_size, 1      , 1             , attn]
    # trg_mask = [batch_size, 1      , query_specific, attn]

    if mask is not None:
      energy = energy.masked_fill(mask == 0, -1e10)

    attention = self.dropout(F.softmax(energy, dim=-1))
    # attention = [batch_size, n_heads, seq_size, seq_size]

    x = torch.einsum('bhjd,bhij->bhid',V,attention)
    # x = torch.matmul(attention, V)
    # x = [batch_size, n_heads, seq_size, hid_dim // n heads]

    x = x.permute(0, 2, 1, 3).contiguous()
    # x = [batch_size, seq_size, n_heads, hid_dim // n heads]

    x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
    # x = [batch_size, src_seq_size, hid_dim]

    x = self.linear(x)
    # x = [batch_size, seq_size, hid_dim]

    return x, attention.detach()

  def reset_parameters(self):
    # nn.init.xavier_normal_(self.w_q.weight)
    # nn.init.xavier_normal_(self.w_k.weight)
    # nn.init.xavier_normal_(self.w_v.weight)
    # nn.init.xavier_normal_(self.linear.weight)
    nn.init.xavier_uniform_(self.w_q.weight)
    nn.init.xavier_uniform_(self.w_k.weight)
    nn.init.xavier_uniform_(self.w_v.weight)
    nn.init.xavier_uniform_(self.linear.weight)


class PositionwiseFeedforward(nn.Module):
  def __init__(self, hid_dim, pf_dim, dropout):
    super().__init__()

    self.hid_dim = hid_dim
    self.pf_dim = pf_dim

    self.linear1 = nn.Linear(hid_dim, pf_dim)
    self.linear2 = nn.Linear(pf_dim, hid_dim)
    self.dropout = nn.Dropout(dropout)

    self.reset_parameters()

  def forward(self, x):
    # x = [batch_size, seq_size, hid_dim]

    x = self.linear1(x)
    x = self.dropout(F.relu(x))
    x = self.linear2(x)

    # x = [batch_size, seq_size, hid_dim]
    return x

  def reset_parameters(self):
    #nn.init.kaiming_normal_(self.linear1.weight, a=math.sqrt(5))
    #nn.init.xavier_normal_(self.linear2.weight)
    nn.init.xavier_uniform_(self.linear1.weight)
    nn.init.xavier_uniform_(self.linear2.weight)


class Decoder(nn.Module):
  def __init__(self, hid_dim, n_layers, n_heads, pf_dim, decoder_layer,
               self_attention, positionwise_feedforward, dropout):
    super().__init__()

    self.layers = nn.ModuleList([decoder_layer(hid_dim, n_heads, pf_dim,
                                               self_attention,
                                               positionwise_feedforward,
                                               dropout)
                                 for _ in range(n_layers)])

  def forward(self, trg, src, trg_mask, src_mask):
    # trg = [batch_size, trg_seq_size, hid_dim]
    # src = [batch_size, src_seq_size, hid_dim]
    # trg_mask = [batch_size, trg_seq_size]
    # src_mask = [batch_size, src_seq_size]
    for layer in self.layers:
      trg = layer(trg, src, trg_mask, src_mask)

    return trg


class DecoderLayer(nn.Module):
  def __init__(self, hid_dim, n_heads, pf_dim, self_attention,
               positionwise_feedforward, dropout):
    super().__init__()

    self.layernorm1 = nn.LayerNorm(hid_dim)
    self.layernorm2 = nn.LayerNorm(hid_dim)
    self.layernorm3 = nn.LayerNorm(hid_dim)
    self.layernorm4 = nn.LayerNorm(hid_dim)
    self.selfAttn = self_attention(hid_dim, n_heads, dropout)
    self.encAttn = self_attention(hid_dim, n_heads, dropout)
    self.densefilter = positionwise_feedforward(hid_dim, pf_dim, dropout)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.dropout3 = nn.Dropout(dropout)

  def forward(self, trg, src, trg_mask, src_mask):
    # trg = [batch_size, trg_seq_size, hid_dim]
    # src = [batch_size, src_seq_size, hid_dim]
    # trg_mask = [batch_size, trg_seq_size]
    # src_mask = [batch_size, src_seq_size]

    # self attention
    z = self.layernorm1(trg)
    z, attn = self.selfAttn(z, z, z, trg_mask)
    z = self.dropout1(z)
    trg = trg + z

    # encoder attention
    z = self.layernorm2(trg)
    z, attn = self.encAttn(z, src, src, src_mask)
    z = self.dropout2(z)
    trg = trg + z

    # dense filter
    z = self.layernorm3(trg)
    z = self.densefilter(z)
    z = self.dropout3(z)
    trg = trg + z

    return self.layernorm4(trg)


class Seq2Seq(nn.Module):
  def __init__(self, embedding, encoder, decoder, pad_idx):
    super().__init__()

    self.embedding = embedding
    self.encoder = encoder
    self.decoder = decoder
    self.pad_idx = pad_idx

  def make_masks(self, src, trg):
    # src = [batch_size, src_seq_size]
    # trg = [batch_size, trg_seq_size]

    src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
    trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)
    # trg_mask = [batch_size, 1, trg_seq_size, 1]
    trg_len = trg.shape[1]

    trg_sub_mask = torch.tril(
      torch.ones((trg_len, trg_len), dtype=torch.uint8, device=trg.device))

    trg_mask = trg_pad_mask & trg_sub_mask

    # src_mask = [batch_size, 1, 1, pad_seq]
    # trg_mask = [batch_size, 1, pad_seq, past_seq]
    return src_mask, trg_mask

  def forward(self, src, trg):
    # src = [batch_size, src_seq_size]
    # trg = [batch_size, trg_seq_size]

    src_mask, trg_mask = self.make_masks(src, trg)
    # src_mask = [batch_size, 1, 1, pad_seq]
    # trg_mask = [batch_size, 1, pad_seq, past_seq]

    src = self.embedding(src)
    trg = self.embedding(trg)
    # src = [batch_size, src_seq_size, hid_dim]

    enc_src = self.encoder(src, src_mask)
    # enc_src = [batch_size, src_seq_size, hid_dim]

    out = self.decoder(trg, enc_src, trg_mask, src_mask)
    # out = [batch_size, trg_seq_size, hid_dim]

    logits = self.embedding.transpose_forward(out)
    # logits = [batch_size, trg_seq_size, d_vocab]

    return logits


  def make_src_mask(self, src):
    # src = [batch size, src sent len]
    src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask

  def make_trg_mask(self, trg):
    # trg = [batch size, trg sent len]
    trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)

    trg_len = trg.shape[1]

    trg_sub_mask = torch.tril(
      torch.ones((trg_len, trg_len), dtype=torch.uint8, device=trg.device))

    trg_mask = trg_pad_mask & trg_sub_mask

    return trg_mask

  def greedy_inference(model, src, sos_idx, eos_idx, max_length, device):
    model.eval()
    src = src.to(device)
    src_mask = model.make_src_mask(src)
    src_emb = model.embedding(src)

    # run encoder
    enc_src = model.encoder(src_emb, src_mask)
    trg = torch.ones(src.shape[0], 1).fill_(sos_idx).type_as(src).to(device)

    done = torch.zeros(src.shape[0]).type(torch.uint8).to(device)
    for _ in range(max_length):
      trg_emb = model.embedding(trg)
      trg_mask = model.make_trg_mask(trg)
      # run decoder
      output = model.decoder(src=enc_src, trg=trg_emb,
                             src_mask=src_mask, trg_mask=trg_mask)
      logits = model.embedding.transpose_forward(output)
      pred = torch.argmax(logits[:,[-1],:], dim=-1)
      trg = torch.cat([trg, pred], dim=1)

      eos_match = (pred.squeeze(1) == eos_idx)
      done = done | eos_match

      if done.sum() == src.shape[0]:
        break

    return trg