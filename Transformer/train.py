import torch
import torch.nn.functional as F
import torch.optim as optim
from transformer import transformer, get_seq_mask
from transformer_optimizer import ScheduledOptim
from pyitcast.transformer_utils import Batch, LabelSmoothing
import numpy as np
from torch.autograd import Variable
from LabelSmooth import LabelSmoothLoss


def data_generator(V, batch_size, num_batch):
    for i in range(num_batch):
        data = torch.from_numpy(np.random.randint(0, V, size=(batch_size, 10)))
        source = Variable(data, requires_grad=False)
        target = Variable(data, requires_grad=False)
        yield Batch(source, target)  # batch生成器


def cal_loss(pred, src, smoothing=False):
    src = src.contiguous().view(-1)
    if smoothing:
        Labelsmoothing = LabelSmoothLoss(0.1)
        loss = Labelsmoothing(src, pred)
    else:
        loss = F.cross_entropy(pred, src.long(), reduction='sum')

    pred = pred.max(1)[1]
    n_correct = pred.eq(src).sum().item()
    n_word = src.size(0)

    return loss, n_correct, n_word


def train_epoch(model, training_data, optimizer, smoothing=False):
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Training)   '
    print(desc)
    for i, batch in enumerate(training_data):
        src_seq, trg_seq = batch.src, batch.trg  # [32,10]
        optimizer.zero_grad()  # 清空过往梯度
        pred = model(src_seq, trg_seq, src_seq_mask=None, trg_seq_mask=get_seq_mask(trg_seq))  # [320,11]
        loss, n_correct, n_word = cal_loss(pred, src_seq, smoothing=smoothing)  # 计算loss
        loss.backward()  # 计算梯度
        optimizer.step_and_update()  # 更新学习率，进行梯度更新
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

    loss = total_loss * 1.0 / n_word_total
    accuracy = n_word_correct * 1.0 / n_word_total
    print("word_total:", n_word_total, ", correct_word:", n_word_correct)
    return loss, accuracy


def eval_epoch(model, validation_data, smoothing=False):
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Validation) '
    print(desc)
    with torch.no_grad():  # 在该模块下的所有tensor,requires_grad=False
        for i, batch in enumerate(validation_data):
            src_seq, trg_seq = batch.src, batch.src  # [32,10]
            pred = model(src_seq, trg_seq, src_seq_mask=None, trg_seq_mask=get_seq_mask(trg_seq))  # [320,11]
            loss, n_correct, n_word = cal_loss(pred, src_seq, smoothing=smoothing)
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

    loss = total_loss * 1.0 / n_word_total
    accuracy = n_word_correct * 1.0 / n_word_total
    print("word_total:", n_word_total, ", correct_word:", n_word_correct)
    return loss, accuracy


def train(model, optimizer, epochs):
    for epoch_i in range(epochs):
        training_data, validation_data = data_generator(11, 32, 20), data_generator(11, 32, 10)

        model.train()  # 模型正处于训练阶段
        train_loss, train_accu = train_epoch(model, training_data, optimizer, smoothing=True)
        lr = optimizer._optimizer.param_groups[0]['lr']
        print("epoch ", epoch_i, "    train_loss:", train_loss, ",train_accu:", train_accu, ",lr:", lr)

        model.eval()  # 模型处于测试阶段
        valid_loss, valid_accu = eval_epoch(model, validation_data, smoothing=False)
        print("epoch ", epoch_i, "    valid_loss:", valid_loss, ",valid_accu:", valid_accu, ",lr:", lr)

    model.eval()
    test = torch.tensor([[1, 3, 2, 4, 6]])
    result = model(test, test, trg_seq_mask=get_seq_mask(test))
    print(result.max(1)[1])


def main():
    model_transformer = transformer(
        n_trg_vocab=11,
        n_layers=8,
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

    optimizer = ScheduledOptim(
        optimizer=optim.Adam(model_transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        lr=2,
        d_model=512,
        n_warm_steps=4000)

    train(model_transformer, optimizer, 1)


if __name__ == '__main__':
    main()
