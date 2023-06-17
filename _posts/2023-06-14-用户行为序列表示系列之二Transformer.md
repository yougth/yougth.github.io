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

1. Embedding	  Item Feature: Ei=Embedding(Item);
2. self-Attention  $$Attention(Q, K, V)= softmax(QK^T/d^{1/2})V $$
3. Multi-Head Attention $$ MultiHead(Q, K, V)=Concat(head1, head2, headh)W^H $$,	$$ where  head=Attention(EW^E, EW^K, EW^V) $$
4. Position-wise Feed-Forward Networks 	FFN(x) = max(0,xW1+b1)W2+b2, where x= MultiHead(Q, K, V)

加入target item之后：

1. Embedding Item_Feature: Ei=Embedding(Item) E=Ei
2. Target Attention $$Interest = Attention(Q, K, V)= softmax(QK^T/d^{1/2})V$$	$$where K= EW^K, V=EW^V, E=Ei, Ei=FFN(x)(上一阶段行为序列建模输入)$$;	$$Q=EqW^Q 为target item的表示W矩阵$$

### 引入更多信息

1. Item Feature: Ei=Embedding(Item);
2. Time Feature: Et=Embedding(ceil(log2(T_{request}-T_{click}))), 其中T_{request}表示当前请求时间，而T_{click}表示用户点击时候时间。
3. Positional Feature: Ep=Embedding(Rank(Trequest-Tclick));
4. Dwell Time Feature: Ed=Embedding(ceil(log2T_{dwell})),T_{dwell}表示用户点击停留时长。
5. Click Source Feature: Es=Embedding(Click Source),Click Source 表示点击的来源，比如特定推荐位。
6. Click Count Feature: Ec=Embedding(Click Count),点击次数
7. E=Ei+Et+Ep+Ed+Es+Ec

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


class SelfAttention(Attention):

    def __call__(self, x, bias, cache=None):
        return super(SelfAttention, self).__call__(x, x, bias, cache)


class FeedFowardNetwork(Module):

    def __init__(self, name, hidden_size,
            filter_size, relu_dropout, allow_pad):
        super(FeedFowardNetwork, self).__init__(name)
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout
        self.allow_pad = allow_pad

        with self.name_scope:
            self.filter_dense_layer = tf.layers.Dense(filter_size,
                use_bias=True, activation=tf.nn.relu, name="filter_layer")

            self.output_dense_layer = tf.layers.Dense(hidden_size,
                use_bias=True, name="output_layer")

    @Module.with_name_scope
    def __call__(self, x, padding=None):
        padding = None if not self.allow_pad else padding

        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]
        if padding is not None:
            with tf.name_scope("remove_padding"):
                # Flatten padding to [batch_size*length]
                pad_mask = tf.reshape(padding, [-1])
                nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))

                # Reshape x to [batch_size*length, hidden_size] to remove padding
                x = tf.reshape(x, [-1, self.hidden_size])
                x = tf.gather_nd(x, indices=nonpad_ids)

                # Reshape x from 2 dimensions to 3 dimensions.
                x.set_shape([None, self.hidden_size])
                x = tf.expand_dims(x, axis=0)
        output = self.filter_dense_layer(x)
        #TODO dropout
        output = self.output_dense_layer(output)
        if padding is not None:
            with tf.name_scope("re_add_padding"):
                output = tf.squeeze(output, axis=0)
                output = tf.scatter_nd(
                    indices=nonpad_ids,
                    updates=output,
                    shape=[batch_size * length, self.hidden_size])
                output = tf.reshape(output, [batch_size, length, self.hidden_size])

        return output


class LayerNormalization(Module):

    def __init__(self, name, hidden_size, epsilon=1e-6):
        super(LayerNormalization, self).__init__(name)
        self.hidden_size = hidden_size
        self.epsilon=epsilon
        with self.name_scope:
            self.scale = tf.Variable(
                tf.ones([self.hidden_size]),
                name='scale')
            self.bias = tf.Variable(
                tf.zeros([self.hidden_size]),
                name='bias')

    @Module.with_name_scope
    def __call__(self, x):
        mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + self.epsilon)
        return norm_x * self.scale + self.bias


class PrePostProcessingWrapper(Module):

    def __init__(self, name, module, hidden_size,
            use_mult=False, norm_first=True):
        super(PrePostProcessingWrapper, self).__init__(name)
        self.use_mult = use_mult
        self.norm_first = norm_first
        self.module = module
        with self.name_scope:
            self.layer_norm = LayerNormalization(
                'layer_norm',
                hidden_size)

    @Module.with_name_scope
    def __call__(self, x, *args, **kwargs):
        if self.norm_first:
            y = self.layer_norm(x)
            y = self.module(y, *args, **kwargs)
            #TODO postprocessing dropout
            if self.use_mult:
                return x + y + x*y
            else:
                return x + y
        else:
            y = self.module(x, *args, **kwargs)
            if self.use_mult:
                y = x + y + x*y
            else:
                y = x + y
            return self.layer_norm(y)


class EncoderStack(Module):

    def __init__(self,
            name,
            transformer_params,
            ):
        super(EncoderStack, self).__init__(name)
        self.transformer_params = transformer_params
        self.att_layers = []
        self.ff_layers = []

        with self.name_scope:
            for i in range(transformer_params['num_hidden_layers']):
                att_layer = SelfAttention(
                    'self_attention_%d' % (i,),
                    transformer_params['hidden_size'],
                    transformer_params['hidden_size_inner'],
                    transformer_params['num_heads'],
                    transformer_params['attention_dropout']
                    )
                ff_layer = FeedFowardNetwork(
                    'feed_forward_%d' % (i,),
                    transformer_params['hidden_size'],
                    transformer_params['filter_size'],
                    transformer_params['relu_dropout'],
                    allow_pad=True
                    )
                wrapped_att_layer = PrePostProcessingWrapper(
                    'att_wrapper_%d' % (i,),
                    att_layer,
                    transformer_params['hidden_size'],
                    norm_first=False
                    )
                wrapped_ff_layer = PrePostProcessingWrapper(
                    'ff_wrapper_%d' % (i,),
                    ff_layer,
                    transformer_params['hidden_size'],
                    norm_first=False
                    )

                self.att_layers.append(wrapped_att_layer)
                self.ff_layers.append(wrapped_ff_layer)

    @Module.with_name_scope
    def __call__(self, x, attention_bias, inputs_padding):
        for i in range(self.transformer_params['num_hidden_layers']):
            att_out = self.att_layers[i](x, attention_bias)
            ff_out = self.ff_layers[i](att_out, inputs_padding)
        return ff_out


class Decoder(Module):

    def __init__(self, name, transformer_params, **kwargs):
        super(Decoder, self).__init__(name)
        self.transformer_params = transformer_params
        with self.name_scope:
            self.input_dense_layer = tf.keras.layers.Dense(
                transformer_params['hidden_size'],
                use_bias=False,
                name="input_transform")
            att_layer = Attention(
                'attention',
                transformer_params['hidden_size'],
                transformer_params['hidden_size_inner'],
                transformer_params['num_heads'],
                transformer_params['attention_dropout']
                )
            ff_layer = FeedFowardNetwork(
                'feed_forward',
                transformer_params['hidden_size'],
                transformer_params['filter_size'],
                transformer_params['relu_dropout'],
                allow_pad=False
                )
            self.wrapped_att_layer = PrePostProcessingWrapper(
                'att_wrapper',
                att_layer,
                transformer_params['hidden_size'],
                use_mult=True
                )
            self.wrapped_ff_layer = PrePostProcessingWrapper(
                'ff_wrapper',
                ff_layer,
                transformer_params['hidden_size'],
                )


    @Module.with_name_scope
    def __call__(self, q, kv, attention_bias):
        # attention_bias = model_utils.get_padding_bias(mask)
        q = self.input_dense_layer(q)
        x = self.wrapped_att_layer(q, kv, attention_bias)
        output = self.wrapped_ff_layer(x)
        # output_normalization = self.output_bn(x, training)
        # output = self.output_flatten(output_normalization)
        return output


class Transformer(Module):

    def __init__(self,
            name,
            transformer_params,
            session_sparse_conf,
            query_size,
            query_fc_layers=[128, 64],
            serve=False,
            ):
        super(Transformer, self).__init__(name)
        self.params = transformer_params
        self.serve = serve
        self.session_type_num \
            = len(session_sparse_conf['namespace_conf']['keeps_conf'])
        self.name = name

        with self.name_scope:
            self.action_embedding = lf.struct.LookupSparse(
                self.name + 'action_embedding',
                emb_dim=session_sparse_conf['emb_dim'],
                space_size=session_sparse_conf['hash_size'] + 1,
                part_num=session_sparse_conf['emb_part_num'],
                combiner=session_sparse_conf['combiner'],
                )
            self.encoder_stack = EncoderStack(
                self.name + 'encoder_stack',
                transformer_params
                )
            self.decoder = Decoder(
                self.name + 'decoder',
                transformer_params
                )

            self.output_bn = tf.keras.layers.BatchNormalization(name= self.name + "output_transform")
            self.output_flatten = tf.keras.layers.Flatten()
            self.output_transform = tf.keras.layers.Dense(
                query_fc_layers[-1], use_bias=False, name= self.name + "output_transform")

    def embedding_encoder(self, act_flat_sparse, indexes):

        act_flat_emb = self.action_embedding(act_flat_sparse)
        act_emb = tf.reshape(act_flat_emb, [-1, self.params['hidden_size']])
        act_fea = tf.concat([
            tf.zeros(
                [1, self.params["hidden_size"]],
                dtype=tf.float32),
            act_emb], axis=0)
        sessions = tf.nn.embedding_lookup(
            act_fea,
            indexes,
            name='padding_session')

        return sessions

    def embedding_decoder(self, encoder_output, encoder_indexes, decoder_indexes):
        encoded_embedding = tf.boolean_mask(
            encoder_output,
            tf.cast(encoder_indexes, tf.bool)
            )
        encoded_embedding = tf.concat(
            [
                tf.zeros(
                    [1, self.params["hidden_size"]],
                    dtype=tf.float32
                    ),
                encoded_embedding
            ], 0)
        decoder_embedding = tf.nn.embedding_lookup(
            encoded_embedding,
            decoder_indexes)
        return decoder_embedding

    def encode(self, inputs, indexes):
        with tf.name_scope(self.name + "encode"):
            embedded_inputs = inputs
            attention_bias = model_utils.get_padding_bias(indexes)
            inputs_padding = model_utils.get_padding(indexes)

            with tf.name_scope(self.name + "add_pos_encoding"):
                length = tf.shape(embedded_inputs)[1]
                pos_encoding = model_utils.get_position_encoding(
                    length, self.params["hidden_size"])

                pos_encoding_revert = model_utils.revert_pos_encoding(
                    pos_encoding, indexes)
                encoder_inputs = embedded_inputs + pos_encoding_revert

        return self.encoder_stack(
            encoder_inputs, attention_bias, inputs_padding)

    def decode(self, q, kv, kv_indexes):
        with tf.name_scope(self.name + "decode"):
            attention_bias = model_utils.get_padding_bias(kv_indexes)

            with tf.name_scope(self.name + "add_pos_encoding"):
                length = tf.shape(kv)[1]
                pos_encoding = model_utils.get_position_encoding(
                    length, self.params["hidden_size"])

                pos_encoding_revert = model_utils.revert_pos_encoding(
                    pos_encoding, kv_indexes)
                kv = kv + pos_encoding_revert

        return self.decoder(q, kv, attention_bias)

    @Module.with_name_scope
    def __call__(self, query,
            act_flat_sparse, sess_indexes, act_indexes,
            training):

        sess_indexes = tf.reshape(sess_indexes, [tf.shape(sess_indexes)[0], -1])

        sessions = self.embedding_encoder(act_flat_sparse, sess_indexes)

        encoded_sessions = self.encode(sessions, sess_indexes)

        action_indexes = tf.reshape(act_indexes, [tf.shape(act_indexes)[0], -1])
        encoded_actions = self.embedding_decoder(
            encoded_sessions,
            sess_indexes,
            action_indexes)

        if self.serve:
            query = tf.expand_dims(query, 0)
        else:
            query = tf.expand_dims(query, 1)
        output = self.decode(query, encoded_actions, action_indexes)
        output = self.output_bn(output, training)
        output = self.output_transform(output)
        if self.serve:
            output = tf.squeeze(output, [0])
        output = self.output_flatten(output)

        return output

```

### 总结

1. 本身设计很灵活，可以各种灵活套用
2. 建模序列长度很友好，因为每个节点都把序列上的其他节点看做无差的做attention，所以不存在长度边长后信息消失，比如我们使用150长度的序列建模
3. 并行能力强，引入矩阵运算，计算快且并行能力强
4. 模型表示能力强大，q可以理解全要查询的query（苹果手机），k理解为查询分解出来的key(apple品牌（k-v 100%）, 苹果水果（k-v 0%）)，而v则是实际查询出来的结果value(iphone)

