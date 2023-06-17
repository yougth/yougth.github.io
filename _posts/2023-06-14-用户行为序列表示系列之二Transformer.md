---
layout:     post
title:      用户行为序列表示系列之二transformer模型
subtitle:   行为序列复杂模型建模
date:       2021-06-19
author:     BY
header-img: img/post-bg-universe.jpg
catalog: true
tags:
    - 行为序列
    - 模型
    - transformer
---

### 背景

用户行为序列建模一路从DIN、DIEN、SIM、到transformer走来，现在基本形成了transformer一统天下的格局。

transfromer是google在2017年[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)提出，用来解决文本的序列建模的模型，后来逐渐替代了并行计算能力差的RNN和无法捕捉远距离信息的CNN，成为了文本特征提取器一统天下的baseline，后来逐渐被搜广推的ctr二分类模型和CV借鉴。

Transformer能一统天下主要是因为特征抽取能力强，长距离特征捕获能力强，并行计算能力好。

### 模型

模型细节比较多，[这里](https://luweikxy.gitbook.io/machine-learning-notes/self-attention-and-transformer)有一篇讲的不错的。

当然原始的transformer和在行为序列中使用的是有一些变化的。

encode本身和原始一样，直接使用行为序列中item做self attention堆叠，而decode需要引入target item的embedding做key，从而检索序列中的匹配信息。

最终对原始的encode的信息和decode的信息做concat作为最终的行为序列建模结果。

行为序列建模表示为

 > A. Embedding	  Item Feature: Ei=Embedding(Item);
 > B. self-Attention  $$Attention(Q, K, V)= softmax(QK^T/d^{1/2})V $$
 > C. Multi-Head Attention $$ MultiHead(Q, K, V)=Concat(head1, head2, headh)W^H, where head=Attention(EW^E, EW^K, EW^V) $$
 > D. Position-wise Feed-Forward Networks 	FFN(x) = max(0,xW1+b1)W2+b2, where x= MultiHead(Q, K, V)

加入target item之后：

 > A. Embedding 	Item_Feature: Ei=Embedding(Item) E=Ei
 > B. Target Attention	$$Interest = Attention(Q, K, V)= softmax(QK^T/d^{1/2})V where K= EW^K, V=EW^V, E=Ei, Ei=FFN(x)(上一阶段行为序列建模输入)；  Q=EqW^Q 为target item的表示W矩阵$$

### 引入更多信息

Item Feature: Ei=Embedding(Item);

Time Feature: Et=Embedding(ceil(log2(T_{request}-T_{click}))), 其中T_{request}表示当前请求时间，而T_{click}表示用户点击时候时间。

Positional Feature: Ep=Embedding(Rank(Trequest-Tclick));

Dwell Time Feature: Ed=Embedding(ceil(log2T_{dwell})),T_{dwell}表示用户点击停留时长。

Click Source Feature: Es=Embedding(Click Source),Click Source 表示点击的来源，比如特定推荐位。

Click Count Feature: Ec=Embedding(Click Count),点击次数

E=Ei+Et+Ep+Ed+Es+Ec

序列长度为150，直接引入以上信息作为对行为序列中item的补充信息建模，在引入target items之后固定时候feture emb固定表示即可，

### 实现

```python
class Attention(Module):

    def __init__(self, name, hidden_size, hidden_size_inner, \
            num_heads, attention_dropout):
        super(Attention, self).__init__(name)
        self.hidden_size = hidden_size
        self.hidden_size_inner = hidden_size_inner
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout

        with self.name_scope:
            self.q_dense_layer = tf.keras.layers.Dense(
                hidden_size_inner, use_bias=False, name="q")

            self.k_dense_layer = tf.keras.layers.Dense(
                hidden_size_inner, use_bias=False, name="k")

            self.v_dense_layer = tf.keras.layers.Dense(
                hidden_size_inner, use_bias=False, name="v")

            self.depth = (self.hidden_size_inner // self.num_heads)
            self.output_dense_layer = tf.keras.layers.Dense(
                hidden_size, use_bias=False, name="output_transform")

    def split_heads(self, x):
        with tf.name_scope("split_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]
            x = tf.reshape(x, [batch_size, length, self.num_heads, self.depth])
        return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]
            x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [batch_size, length, self.hidden_size_inner])

    @Module.with_name_scope
    def __call__(self, x, y, bias, cache=None):
        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)
        if cache is not None:
            # Combine cached keys and values with new keys and values.
            k = tf.concat([cache["k"], k], axis=1)

            # Update cache
            cache["k"] = k
            cache["v"] = v

        q = self.split_heads(q)   # (batch, num_heads, len_x, hidden_size/num_heads)
        k = self.split_heads(k)
        v = self.split_heads(v)

        q *= self.depth ** -0.5

        logits = tf.matmul(q, k, transpose_b = True)
        logits += bias # attention bias and position bias
        weights = tf.nn.softmax(logits, name="attention_weights")
#        if self.train:
#        weights = tf.nn.dropout(weights, 1.0 - self.attention_dropout)
        attention_output = tf.matmul(weights, v)

        attention_output = self.combine_heads(attention_output)
        attention_output = self.output_dense_layer(attention_output)
        return attention_output
```

### 总结

