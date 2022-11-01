import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def get_seq_mask(seq):
    seq_mask = (1 - torch.triu(torch.ones(1, seq.shape[1], seq.shape[1]), diagonal=1))
    return seq_mask


# --------------------------------实现单词编码-------------------------------------------------------
## 实现单词原始编码
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


## 实现单词的位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        PE = torch.zeros(max_len, d_model)  # [max_len,d_model]
        for pos in torch.arange(0, max_len):
            PE[pos:, 0::2] = torch.sin(pos / torch.pow(10000.0, torch.arange(0, d_model, 2) / d_model))  # dim 2i
            PE[pos:, 1::2] = torch.cos(pos / torch.pow(10000.0, torch.arange(0, d_model, 2) / d_model))  # dim 2i+1

        PE = PE.unsqueeze(0)  # [1,max_len,d_model]
        self.register_buffer('PE',
                             PE)  # This is typically used to register a buffer that should not to be considered a model parameter.

    def forward(self, x):
        x = x + Variable(self.PE[:, :x.size(1), :], requires_grad=False)
        return self.dropout(x)


'''
d_model = 512
vocab = 1000
dropout = 0.1
max_len = 60
x = Variable(torch.LongTensor([[1, 2, 3, 4], [5, 6, 7, 8]]))  # [2,4]
emb = Embeddings(d_model, vocab)
embr = emb(x)
print(embr)  # [2,4,512]
pe = PositionalEncoding(d_model, dropout, max_len)
pe_result = pe(embr)
print(pe_result)  # [2,4,512]
'''


# -------------------------------实现多头注意力机制--------------------------------------------
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, attn_dropout=0):
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / math.sqrt(self.d_model), k.transpose(2, 3))  # O(d * n^2)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e6)
        # print(attn)
        attn = self.dropout(F.softmax(attn, dim=-1))  # O(n^2)
        output = torch.matmul(attn, v)  # O(d * n^2)
        return output, attn  # 总时间复杂度O(d * n^2)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_q = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.fc1 = nn.Linear(d_model, n_head * d_model, bias=False)
        self.fc2 = nn.Linear(d_k, n_head * d_k, bias=False)
        self.fc3 = nn.Linear(d_v, n_head * d_v, bias=False)

        self.fc4 = nn.Linear(n_head * d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)  # 只考虑最低值

    def forward(self, q, k, v, mask=None):
        d_q, d_k, d_v, n_head = self.d_q, self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q

        q = self.fc1(q).view(sz_b, len_q, n_head, d_q)
        k = self.fc2(k).view(sz_b, len_k, n_head, d_k)
        v = self.fc3(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [sz_b, n_head, len_q, d_q]
        if mask is not None:
            mask = mask.unsqueeze(1)

        output, attn = self.attention(q, k, v, mask=mask)  # [sz_b, n_head, len_q, d_q]
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q,
                                                          -1)  # concatenate，transpose致使张量在内存中不连续，view 函数只能用于内存中连续存储的tensor

        output = self.dropout(self.fc4(output)) + residual
        output = self.layer_norm(output)

        return output, attn


'''
n_head = 8
sd = ScaledDotProductAttention(512)
mask = get_seq_mask(pe_result)
print(mask)
multihead = MultiHeadAttention(n_head, d_model, d_model, d_model)
output, attn = multihead(pe_result, pe_result, pe_result,mask=mask)
print(output)
print(output.shape)
'''


# ------------------------------实现前馈全连接层-----------------------------------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_inner)
        self.fc2 = nn.Linear(d_inner, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.fc2(F.relu(self.fc1(x)))
        x = self.dropout(x) + residual
        x = self.layer_norm(x)

        return x


'''
FeedForward = PositionwiseFeedForward(d_model, d_model * 4)
ff = FeedForward(output)
print(ff)
print(ff.shape)
'''


# --------------------------实现编码器层和解码器层-----------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_input, mask=None):
        enc_output, enc_attn = self.attn(enc_input, enc_input, enc_input, mask=mask)
        enc_output = self.ffn(enc_output)

        return enc_output, enc_attn


'''
el = EncoderLayer(d_model, 4 * d_model, n_head, d_model, d_model)
encoder_out, enc_attn = el(pe_result)
print(encoder_out)
print(encoder_out.shape)
print(enc_attn)
print(enc_attn.shape)
'''


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)  # masked self-attention,Q=K=V
        self.attn2 = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)  # cross-attention,Q!=K=V
        self.ffn = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, dec_input, enc_out, self_attn_mask=None, dec_enc_attn_mask=None):
        dec_out, dec_self_attn = self.attn1(dec_input, dec_input, dec_input, mask=self_attn_mask)
        dec_out, dec_enc_attn = self.attn2(dec_out, enc_out, enc_out, mask=dec_enc_attn_mask)
        dec_out = self.ffn(dec_out)

        return dec_out, dec_self_attn, dec_enc_attn


'''
dl = DecoderLayer(d_model, 4 * d_model, n_head, d_model, d_model)
dec_out, dec_self_attn, dec_enc_attn = dl(pe_result, encoder_out)
print(dec_out, dec_self_attn, dec_enc_attn)
'''


# ----------------------------实现编码器和解码器------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, vocab, d_inner, n_head, d_k, d_v, max_len=200, dropout=0.1):
        super(Encoder, self).__init__()
        self.src_word_emb = Embeddings(d_model, vocab)
        self.position_emb = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)
        self.stack_layer = nn.ModuleList(
            [EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seg, src_mask=None, return_attn=False):
        enc_self_attn_list = []
        enc_output = self.src_word_emb(src_seg)
        enc_output = self.position_emb(enc_output)

        for enc_layer in self.stack_layer:
            enc_output, enc_attn = enc_layer(enc_output, mask=src_mask)
            enc_self_attn_list += [enc_attn] if return_attn else []

            if return_attn:
                return enc_output, enc_self_attn_list

        return enc_output


'''
encoder = Encoder(8, d_model, vocab, 4 * d_model, 8, d_model, d_model)
encoder_out = encoder(x)
print(encoder_out)
print(encoder_out.shape)
'''


class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, vocab, d_inner, n_head, d_k, d_v, max_len=200, dropout=0.1):
        super(Decoder, self).__init__()
        self.trg_word_emb = Embeddings(d_model, vocab)
        self.position_emb = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)
        self.stack_layer = nn.ModuleList(
            [DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, trg_seq, enc_out, self_attn_mask=None, dec_enc_attn_mask=None, return_attn=False):
        dec_self_attn_list, dec_enc_attn_list = [], []
        dec_output = self.position_emb(self.trg_word_emb(trg_seq))
        for dec_layer in self.stack_layer:
            dec_output, dec_self_attn, dec_enc_attn = dec_layer(dec_output, enc_out, self_attn_mask=self_attn_mask,
                                                                dec_enc_attn_mask=dec_enc_attn_mask)
            dec_self_attn_list += [dec_self_attn] if return_attn else []
            dec_enc_attn_list += [dec_enc_attn] if return_attn else []

        if return_attn:
            return dec_output, dec_self_attn_list, dec_enc_attn_list

        return dec_output


'''
decoder = Decoder(8, d_model, vocab, 4 * d_model, 8, d_model, d_model)
decoder_out = decoder(x, encoder_out)
print(decoder_out)
print(decoder_out.shape)
'''


# ---------------------------实现transformer--------------------------------------------------------------
class transformer(nn.Module):
    def __init__(self, n_trg_vocab, n_layers, d_model, d_inner, n_head, d_k, d_v, max_len=200, dropout=0.1,
                 trg_emb_pri_weight_sharing=True, emb_src_trg_weight_sharing=True):
        super(transformer, self).__init__()
        self.d_model = d_model
        self.Encoder = Encoder(n_layers, d_model, n_trg_vocab, d_inner, n_head, d_k, d_v, max_len=max_len,
                               dropout=dropout)
        self.Decoder = Decoder(n_layers, d_model, n_trg_vocab, d_inner, n_head, d_k, d_v, max_len=max_len,
                               dropout=dropout)
        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  # 使用xavier初始化防止梯度消失

        if emb_src_trg_weight_sharing:  # share the weight between encoder_embeddings and decoder_embeddings
            self.Encoder.src_word_emb.embedding.weight = self.Decoder.trg_word_emb.embedding.weight

        if trg_emb_pri_weight_sharing:  # share the weight between encoder_embeddings and the last dense.
            self.trg_word_prj.weight = self.Decoder.trg_word_emb.embedding.weight

    def forward(self, src_seq, trg_seq, src_seq_mask=None, trg_seq_mask=None):
        enc_out = self.Encoder(src_seq, src_seq_mask)
        dec_out = self.Decoder(trg_seq, enc_out, src_seq_mask, trg_seq_mask)
        seq_logit = self.trg_word_prj(dec_out)
        seq_logit /= math.sqrt(self.d_model)
        seq_logit = seq_logit.view(-1, seq_logit.size(2))

        return seq_logit


'''
transformer = transformer(10,8,512,4*512,8,512,512)
print(transformer)
out = transformer(x,x)
print(out)
print(torch.argmax(out,dim=-1))
print(out.shape)
'''
