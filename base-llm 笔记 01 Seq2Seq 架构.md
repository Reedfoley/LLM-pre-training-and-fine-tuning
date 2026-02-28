Datawhale 正文详见：[datawhale/base-llm](https://github.com/datawhalechina/base-llm/blob/main/docs/chapter4/10_seq2seq.md)
# Seq2Seq 详解

在自然语言处理中，有一类更为复杂的多对多任务，这类任务的输入序列与输出序列在长度上不相等，且元素之间没有严格的对齐关系，最典型的例子便是**机器翻译**问题。

针对这类问题，Seq2Seq，即序列到序列（Seqence-to-Seqence）架构被提出并取得了重大的成功。

##  从自编码器到 Seq2Seq

要了解 Seq2Seq，我们首先要明白一种更为基础的，同样使用编码器-解码器（Encoder-Decoder）思想的无监督神经网络——自编码器（Autoencoder）。

自编码器由两部分组成：

- 编码器：读取输入数据并将其**压缩**为一个低纬度、紧凑的潜在表示（Latent Representation）。可以视为特征提取或数据压缩。
- 解码器：输入这个潜在表示并将其**重构**回原始的输入数据。

通过压缩和重构，自编码可以学习到数据中的核心特征，常用于降维、特征学习或数据去噪等任务。

## Seq2Seq 整体结构

Seq2Seq 同样由编码器和解码器组成：

- 编码器：将输入序列压缩为固定长度的**上下文向量 (context vector)**，记为 $c \in (1, dim)$，相当于输入序列的“语义概要”。
- 解码器：将 $c$ 作为输入，逐个生成输出序列中的每一个**词元**。

虽然有着类似的结构，但 Seq2Seq 的目的不是重构，而是转换。

![](images/4_1_1.svg)

### 编码器（Encoder）

编码器生成上下文向量 $c \in (1, dim)$ 通常使用**循环神经网络 (RNN)**，如 LSTM。

进入 Encoder 前，输入序列的词元首先通过 Embedding 向量化，$x_1, x_2, x_3, ..., x_T \in (1,dim)$。随后，Encoder 会根据时间步（前后文顺序）读取词元向量，更新自身状态。

- 经典RNN：$h_t = f(h_t-1, x_t)$
- LSTM：$(h_t, c_t) = LSTM((h_{t-1}, c_{t-1}), x_t)$

完成最后一个时间步的更新后，我们可以使用多种方法，如直接使用最后一个时间步的隐藏状态，或将所有时间步的隐藏状态进行处理，得到向量 $c$ 或者元组 $c=(h_T,c_T)$ 作为上下文向量。

这只是最简单的方法，如果有需要，你可以使用更复杂的方法，如双向 RNN，到得到信息更丰富的上下文向量。

### 解码器（Decoder）

解码器的初始状态直接使用编码器的生成的上下文向量，相当于将通过编码器得到的输入序列的“语义概要”传递给解码器。

在第一个时间步，使用一个起始符 `<SOS>` (Start of Sentence) 作为解码器（初始状态下）的第一个输入，生成第一个目标词元 $y_1$。

随后，通过上一步生成的词元和更新后的状态，持续生成新的词元，直到生成一个终止符 `<EOS>` (End of Sentence) 或达到最大预设长度。
$$(h^{\prime}_t, c^{\prime}_t) = \text{LSTM}((h^{\prime}_{t-1}, c^{\prime}_{t-1}), y_{t-1})$$
$$
y_t = Softmax(Linear(h^{\prime}_{t}))
$$
注意，解码器生成序列时是按照时间顺序生成的，所有不应该看到未来的词元，因此不能使用任何双向神经网络。

从深层次看，解码器就是一个语言模型，即**预测下一个词元的模型**，其初始条件为编码器生成的上下位向量。

大语言模型则是**基于自身前缀（提示词)** 的语言模型，即 Decoder-only 架构，这是现代大语言模型一致遵循的训练范式。

### Seq2Seq实现细节

#### 词嵌入层设计

在处理词元时，编码器和解码器都需要通过 `Embedding` 层将其转化为向量。你可以选择两种设计方案：

1. 共享词嵌入层：当编码器和解码器处理的语言（源语言和目标语言）词汇表有大量重叠或是想直接将两种语言合并为一个大的词汇表时，可以选择共享词嵌入层。这样做可以减少模型参数，并让模型学习到两种语言间的潜在联系。
2. 不共享词嵌入层：若源语言和目标语言的词汇表彼此独立，则可以让编码器和解码器各自拥有独立的词嵌入层。

#### 上下文向量

上下文向量理论上是编码器的最后的隐藏状态 $c$ 。但是，对于长序列信息，可能会导致信息在经过多步传递后逐渐“稀释”或“遗忘”。

我们可以通过将第 $t$ 个时间步，将常规的词元输入 $y_{t-1}$ 经过`Embedding`层后得到的向量，与上下文向量 $c$ 进行合并（拼接或相加）。

这种方式理论上可以更好地对抗信息遗忘，但它仍然无法解决更深层次的“对齐”问题（即，在生成某个特定词时，应该重点关注输入的哪个部分）。

这也正是后续注意力机制（Attention Mechanism）诞生的重要原因。

#### 损失函数

在训练过程中，我们会在解码器的每一个时间步输出隐藏状态 $h^{\prime}_t$，经过全连接层和 softmax 函数，输出一个词汇表概率分布向量 $p_t$。

使用交叉熵损失函数，计算 $p_t$ 的对应真是词元 $y_t$ 的损失：$Loss_t = -\log p_t(y_t)$，对整个序列的损失进行累加或平均获得总损失。

通过反向传播算法，根据总损失来更新编码器和解码器的参数。

#### 数据填充与特殊词元

在训练过程中，每个句子的序列长度可能不同，为此，需要将其填充至相同长度。在计算损失时，我们将忽略填充位置的损失。

我们将引入一些特殊词元，来辅助模型处理：

- `<PAD>`：填充符，用于对齐每个序列的长度。
- `<SOS>`或`<GO>`：句子起始符，作为解码器的第一个时间步的输入。
- `<EOS>`：句子终止符，当解码器生成它时，表示句子已完整，可以停止生成。
- `<UNK>`：未知词元，用于替换在训练词汇表中未出现过的词，可以增强模型的鲁棒性。

在训练过程中，我们其实可以使用上一个真实词元预测下一个词元，而不是使用解码器输出的预测值。通过这种方式，模型在每个时间步都能学到从正确的历史信息到下一个正确词元的映射关系。

# 训练与推理模式

## 教师强制

在解码器中，将上一个时间步的预测结果作为下一个时间步的输入的方法称为**自回归模式**，这种方法存在两个问题：

- 收敛缓慢：如果出现错误的预测（尤其在预训练初期更为频繁），那么会持续影响后续的预测，导致误差累积，使得模型很难收敛。
- 难以并行：每个时间步的运行都需要依赖上一时间步，使得训练过程无法并行，效率低下。

为此，可以采用了一种名为**教师强制**的训练策略，每个时间步的输入不再是上一步的预测值，而是使用数据集中的**真实值**。这样，解码器每步都会使用正确的历史信息进行预测，避免了误差累积，同时在训练时可以在时间步上并行（配合掩码策略）。

# PyTorch 代码实现与分析

## 编码器

编码器读取输入序列，输出最终的 **隐藏状态(hieedn)** 和 **细胞状态(cell)** 作为上下文向量赋值给解码器中的初始状态。

``` python
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        # nn.Embedding 是 PyTorch 常用模块，负责将输入序列转为向量表示
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, # 词表大小，即词汇表中不同词元的总数
            embedding_dim=hidden_size # 嵌入维度，即每个词元对应的向量维度
        )
        # nn.LSTM 是长短期记忆网络，用于捕捉序列中的长期依赖关系
        self.rnn = nn.LSTM(
            input_size=hidden_size, # 输入特征维度，需与embedding_dim一致
            hidden_size=hidden_size, # 隐藏层状态维度，决定输出向量大小
            num_layers=num_layers, # LSTM堆叠层数，多层可提取更抽象特征
            batch_first=True, # 输入张量格式为 (batch, seq_len, feature)
            bidirectional=False # LSTM，若为True则变为双向LSTM
        )

    def forward(self, x):
        embedded = self.embedding(x) # (batch_size, seq_length) -> (batch_size, seq_length, hidden_size)
        # 返回最终状态，供解码器使用以初始化其隐藏状态
        # hidden(num_layers, batch_size, hidden_size)
        # cell(num_layers, batch_size, hidden_size)
        _, (hidden, cell) = self.rnn(embedded)
        return hidden, cell
```

## 解码器

在每一个时间步中，解码器接收一个词元和前一步的状态，然后输出预测和新的状态。

``` python
class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        # nn.Linear 是全连接层，负责将LSTM输出映射到词汇表大小，用于词元预测
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, x, hidden, cell):
	    # 将批量的当前时间步的 token 转换为解码器期望输入格式
        x = x.unsqueeze(1) # (batch_size) -> (batch_size, 1)

        embedded = self.embedding(x) # (batch_size, 1) -> (batch_size, 1, hidden_size)
        # 返回当前步的输出和更新后的状态
        # outputs(batch_size, 1, hidden_size)
        # hidden(batch_size, hidden_size)
        # cell(batch_size, hidden_size)
        outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        predictions = self.fc(outputs.squeeze(1)) # (batch_size, 1, hidden_size) -> (batch_size, vocab_size)
        return predictions, hidden, cell
```

## Seq2Seq 包装模块

将编码器和解码器连接起来，并实现**训练**逻辑。

``` python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # 获取批次大小和目标序列长度
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features
		
		# 存储每个时间步的预测结果
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)

        # 第一个输入是 <SOS>
        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            # 保存当前时间步的预测结果
            outputs[:, t, :] = output

            # 决定是否使用 Teacher Forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            # 如果 teacher_force，下一个输入是真实值；否则是模型的预测值
            input = trg[:, t] if teacher_force else top1

        return outputs
```

## 高效的推理实现

将**上一步的输出词元**和**上一步的隐藏状态**传入解码器，进行单步计算，然后用返回的新状态覆盖旧状态。

``` python
    def greedy_decode(self, src, max_len=12, sos_idx=1, eos_idx=2):
        """推理模式下的高效贪心解码。"""
        self.eval()
        with torch.no_grad():
            hidden, cell = self.encoder(src)
            trg_indexes = [sos_idx]
            for _ in range(max_len):
                # 1. 输入只有上一个时刻的词元
                trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(self.device)
                
                # 2. 解码一步，并传入上一步的状态
                output, hidden, cell = self.decoder(trg_tensor, hidden, cell)
                
                # 3. 获取当前步的预测，并更新状态用于下一步
                pred_token = output.argmax(1).item()
                trg_indexes.append(pred_token)
                if pred_token == eos_idx:
                    break
        return trg_indexes
```

## 上下文向量的另一种用法

将上下文向量作为解码器**每个时间步的额外输入**，持续地为解码器提供全局信息。

``` python
class DecoderAlt(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(DecoderAlt, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        # 主要改动 1: RNN的输入维度是 词嵌入+上下文向量
        self.rnn = nn.LSTM(
            input_size=hidden_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, x, hidden_ctx, hidden, cell):
        x = x.unsqueeze(1)
        embedded = self.embedding(x)

        # 主要改动 2: 将上下文向量与当前输入拼接
        # 这里简单地取编码器最后一层的 hidden state 作为上下文代表
        context = hidden_ctx[-1].unsqueeze(1).repeat(1, embedded.shape[1], 1)
        rnn_input = torch.cat((embedded, context), dim=2)

        # 解码器的初始状态 hidden, cell 在第一步可设为零；之后需传递并更新上一步状态
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        predictions = self.fc(outputs.squeeze(1))
        return predictions, hidden, cell
```

# 应用与局限

## 广泛性

本质上定义了一个“将一种数据形态转换为另一种数据形态”的通用范式，所以它的应用远不止于文本到文本的任务。

也可以用在**语音识别（Audio-to-Text）**、**图像描述生成（Image-to-Text）**、**文本到语音（Text-to-Speech, TTS）**、**问答系统（QA）中**。

## 瓶颈

编码器和解码器之间唯一的沟通桥梁就是一个**固定长度**的上下文向量 C。编码器必须将输入序列的所有信息，无论其长短，都压缩到这个向量中。

当输入句子很长时，这个最终的上下文向量 C 依然很难承载全部的语义细节，模型可能会“遗忘”掉句子开头的关键信息，导致生成质量下降。

即使采用**将上下文向量作为解码器每个时间步额外输入**的策略，因为每个时间步输入的都是**同一个** C，模型仍然无法学会有选择性地、有侧重地利用输入信息。

