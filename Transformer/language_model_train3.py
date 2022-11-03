import torch
import torch.nn.functional as F
import torchtext
from transformer import transformer, get_seq_mask
from torchtext.data.utils import get_tokenizer
import time
import math
import copy

device = torch.device("cuda:0")


def get_data():
    TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"), init_token="<sos>", eos_token="<eos>",
                                lower=True)  # 根据语义分割器创建一个语料域
    train_text, val_text, test_text = torchtext.datasets.WikiText2.splits(TEXT)  # 导入wikitext-2数据集并用语料域对其进行split
    # print(len(train_text.examples[0].text),train_text.examples[0].text)
    TEXT.build_vocab(train_text)  # 使用训练集创建一个词表vocab
    # print(TEXT.vocab.__getitem__("<sos>"),TEXT.vocab.__getitem__("<eos>"))# 2,3
    # print(TEXT.vocab.itos)
    ntokens = len(TEXT.vocab.stoi)  # 词汇总数
    train_data = batchify(TEXT, train_text, 20)
    # print(train_data.shape) #[104335,20]
    val_data = batchify(TEXT, val_text, 10)
    # print(val_data.shape) #[21817,10]
    test_data = batchify(TEXT, test_text, 10)
    # print(test_data.shape) #[24621,10]
    return train_data.to(device), val_data.to(device), test_data.to(device), ntokens


def batchify(TEXT, data, batch_size):
    data = TEXT.numericalize([data.examples[0].text])  # 将单词映射成对应的连续数字
    num_batch = data.size(0) // batch_size
    data = data.narrow(0, 0, num_batch * batch_size)  # 将不够一个batch的数据丢弃
    data = data.view(-1, batch_size).contiguous()
    # print(data,data.shape)
    return data


def get_batch(src, i, bptt):
    seq_len = min(bptt, len(src) - 1 - i)
    cat = torch.ones(src.size(0), 1, dtype=torch.int) * 3
    cat = cat.cuda()
    src = torch.cat((src, cat), dim=1).cuda()  # 在这里结尾填上<eos>
    data = src[i:i + seq_len, :src.size(1) - 1]
    target = src[i:i + seq_len, 1:].contiguous().view(-1)
    return data, target


def train(model, optimizer, train_data, epoch, bptt):
    model.train()
    total_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)  # [32,20],[700]
        optimizer.zero_grad()
        output = model(data, data, src_seq_mask=None, trg_seq_mask=get_seq_mask(data).cuda())
        loss = F.cross_entropy(output, targets, reduction='sum')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 使用nn自带的clip_grad_norm_方法进行梯度规范化，防止出现梯度爆炸或消失
        optimizer.step()
        total_loss += loss.item()

        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.9f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} '.format(
                epoch, batch, len(train_data) // bptt, optimizer.param_groups[0]['lr'],
                              elapsed * 1000 / log_interval, cur_loss
            ))

        total_loss = 0.
        start_time = time.time()


def evaluate(eval_model, data_source, bptt):
    eval_model.eval()
    total_loss = 0
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)
            output = eval_model(data, data, src_seq_mask=None, trg_seq_mask=get_seq_mask(data).cuda())
            total_loss += F.cross_entropy(output, targets, reduction='sum').item()
            cur_loss = total_loss / ((data_source.size(0) - 1) / bptt)
    return cur_loss


def get_model(ntokens):
    transformer_model = transformer(
        n_trg_vocab=ntokens,
        n_layers=2,
        d_model=512,
        d_inner=2048,
        n_head=8,
        d_k=512,
        d_v=512,
        max_len=200,
        dropout=0.1,
        trg_emb_pri_weight_sharing=True,
        emb_src_trg_weight_sharing=True
    )
    optimizer = torch.optim.SGD(transformer_model.parameters(), lr=5.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    return transformer_model, optimizer, scheduler


def main():
    train_data, val_data, test_data, ntokens = get_data()
    transformer_model, optimizer, scheduler = get_model(ntokens)
    transformer_model.to(device)
    best_model = None
    best_val_loss = float('inf')
    bptt = 35
    for epoch in range(1, 2):
        epoch_start_time = time.time()
        train(transformer_model, optimizer, train_data, epoch, bptt)
        val_loss = evaluate(transformer_model, val_data, bptt)
        print('-' * 50)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} '.format(
            epoch, (time.time() - epoch_start_time), val_loss
        ))
        print('-' * 50)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(transformer_model)
        scheduler.step()
    test_loss = evaluate(best_model, test_data, bptt)
    print('-' * 90)
    print('| End of training | test loss {:5.2f}'.format(test_loss))
    print('-' * 90)


if __name__ == '__main__':
    main()
