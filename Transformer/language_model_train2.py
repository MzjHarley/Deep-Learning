# 数学计算工具包
import math

# torch相关
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
# torch中经典文本数据集有关的工具包
import torchtext

# torchtext中数据处理工具，该函数用于英文分词
from torchtext.data.utils import get_tokenizer

# 已经构建完成的TransformerModel
from pyitcast.transformer import TransformerModel
# 创建语料域, 语料域是存放语料的数据结构,
# 它的四个参数代表给存放语料（或称作文本）施加的作用.
# 分别为 tokenize,使用get_tokenizer("basic_english")获得一个分割器对象,
# 分割方式按照文本为基础英文进行分割.
# init_token为给文本施加的起始符 <sos>给文本施加的终止符<eos>,
# 最后一个lower为True, 存放的文本字母全部小写.
TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)

# 然后使用torchtext的数据集方法导入数据
# 并切分为训练文本，验证文本，测试文本，并对这些文本施加刚刚创建的语料域
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)


# 将训练集文本数据构建一个vocab对象
# 可以用vocab对象的stoi方法统计文本共包含的不重复词汇总数
TEXT.build_vocab(train_txt)

# 然后选择设备
device = torch.device("cuda")
def batchify(data, bsz):
    """
    该函数用于将文本数据映射成连续数字，并转换指定的样式，指定的样式可参考图片
    :param data: 之前得到的文本数据(train_txt, val_txt, test_txt)
    :param bsz: batch_size，每次模型更新参数的数据量
    :return: 处理之后的数据
    """
    # 先将单词映射成连续对应的数字
    data = TEXT.numericalize([data.examples[0].text])

    # 接着用数据词汇总数除bsz并取整得到一个nbatch代表需要多少次batch后遍历所有数据
    nbatch = data.size(0) // bsz

    # 使用narrow方法对不规整剩余数据进行删除
    # 第一个参数代表横轴删除还是纵轴删除，0为横，1为纵
    # 第二个和第三个参数代表保留开始轴到结束轴的数值，类似于切片
    data = data.narrow(0, 0, nbatch*bsz)

    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


# 用batchify来处理训练数据，验证数据以及测试数据
# 训练数据的bsz
batch_size = 20

# 验证和测试数据（统称为评估数据）的bsz
eval_batch_size = 10

# 获得处理后的数据
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)
# 设置句子最大长度35
bptt = 35

def get_batch(source, i):
    """
    用于获取每个批次合理大小的源数据和目标数据
    :param source: 通过batchify得到的三个data
    :param i: 具体批次次数
    :return: 源数据与目标数据
    """
    # 确定句子长度，应该是bptt和len(source)-1-i的小值
    seq_len = min(bptt, len(source)-1-i)

    # 语言模型训练的源数据的第i批次数据将是batchify结果切片
    data = source[i:i+seq_len]

    # 根据语言模型训练的语料规定，他的目标数据是源数据后移一位
    # 最后目标数据的切片会越界，所以使用view(-1)保证形状正常
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target
# 通过TEXT.vocab.stoi属性获得不重复词汇总数
ntokens = len(TEXT.vocab.stoi)
# 词嵌入大小
emsize = 200
# 前馈全连接层节点数
nhid = 200
# 编码器层数量
nlayers = 2
# 多头注意力机制头数
nhead = 2
# 置0比率
dropout = 0.2

# 将参数输入到模型中
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

# 模型初始化后，接下来进行损失函数和优化方法的选择
# 使用nn自带的交叉熵损失
criterion = nn.CrossEntropyLoss()

# 初始学习率
lr = 5.0

# 优化器选择torch自带的SGD随机梯度下降方法
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# 学习率调整方法，使用torch自带的lr_scheduler，将优化器传入
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
def train(epoch):
    """
    训练函数
    :param epoch:循环次数
    :return: None
    """
    # 模型开启训练模式
    model.train()
    total_loss = 0.
    start_time = time.time()
    # 遍历批次数据
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        # 获取源数据和目标数据
        data, targets = get_batch(train_data, i)
        # 设置初始梯度为0
        optimizer.zero_grad()
        # 装入model得到输出
        output = model(data)
        # 将输出和目标数据传入损失函数对象
        loss = criterion(output.view(-1, ntokens), targets)
        # 反向传播获得总损失
        loss.backward()
        # 使用nn自带的clip_grad_norm_方法进行梯度规范化，防止出现梯度爆炸或消失
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        # 更新模型参数
        optimizer.step()
        # 损失加和
        total_loss += loss.item()
        # 日志打印间隔
        log_interval = 200
        # 如果batch是200的倍数，则打印日志
        if batch % log_interval == 0 and batch > 0:
            # 平均损失
            cur_loss = total_loss / log_interval
            # 需要的时间
            elapsed = time.time() - start_time
            # 打印轮数、当前批次和总批次，当前学习率，训练速度
            # 平均损失，以及困惑度
            # 困惑度是衡量语言模型的重要指标，他是交叉熵平均损失取自然对数的底数
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)
            ))
        # 每个批次结束后，总损失归0
        total_loss = 0
        # 开始时间取当前时间
        start_time = time.time()


def evaluate(eval_model, data_source):
    """
    评估函数
    :param eval_model:每轮训练产生的模型
    :param data_source: 验证或测试数据集
    :return: 平均损失
    """
    # 模型开启评估模式
    eval_model.eval()
    # 损失归零
    total_loss = 0
    # 因为评估模式模型参数不变，所以不进行反向传播
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += criterion(output_flat, targets).item()

            cur_loss = total_loss / ((data_source.size(0) - 1) / bptt)
    return cur_loss
# 初始化最佳验证损失，初始值无穷大
import copy
best_val_loss = float('inf')

# 训练轮数
epochs = 1

# 定义最佳模型变量，初值为None
best_model = None

if __name__ == '__main__':
    for epoch in range(1, epochs + 1):
        # 获得轮数开始时间
        epoch_start_time = time.time()
        # 调用训练函数
        train(epoch)
        # 训练后模型参数发生了变化
        # 将模型和评估数据传入评估函数中
        val_loss = evaluate(model, val_data)
        # 打印每轮的评估日志
        print('-'*50)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
            epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)
        ))
        print('-'*50)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
        # 每轮都会对优化方法的学习率进行调整
        scheduler.step()

    test_loss = evaluate(best_model,test_data)
    print(test_loss)