# MXNET-SMN
## MXNET-SMN

Sequential Matching Network(ACL 2017) reimplemented by MXNET1.0-GPU//but you can always use cpu

用mxnet写的SMN，这篇文章发表在ACL 2017，使用MXNET1.0实现

# What's SMN
## SMN是啥？

SoTA model of multi-turn retrieval-based conversation systems.

You can see the paper[SMN](http://www.aclweb.org/anthology/P/P17/P17-1046.pdf)

SMN是目前最屌的多轮检索对话模型.

如果你是个大佬，请去读一下这篇论文[SMN](http://www.aclweb.org/anthology/P/P17/P17-1046.pdf)，这是我这一年中读过的为数不多的好文章.

它坦诚了SMN在某些方面的失败，对于baseline也作了完整归类，总结，所以ACL的质量还是非常高的。

# How to use?
## 你能不能告诉我怎么用？

> 1 . get data from msra[douban corpus](https://1drv.ms/u/s!AtcxwlQuQjw1jF0bjeaKHEUNwitA)
> 2 . get pre-trained word2vec.model use bash *python3.5 gen_w2v.py train.txt train_vec.model train_vec.vec*
> 3 . get processed data use Process.py(also only support py3)
> 4 . run the model

1. 下载数据 [douban corpus](https://1drv.ms/u/s!AtcxwlQuQjw1jF0bjeaKHEUNwitA)
2. 预训练Embedding矩阵*python3.5 gen_w2v.py train.txt train_vec.model train_vec.vec*
3. 预处理数据使用Process.py
4. 跑模型

额外赠礼：每个py文件的作用
|:------------------------------------:|------------------------------------|
|gen_w2v.py|用来生成预训练的词向量|
|Process.py|用来打包数据|
|model.py|用来训练模型|

# Params of Model
## 模型的参数

batch_size = 1000(with 1 titan xp)

embedding_size = 200

gru_layer = 1

max_turn = 10

max_len = 50

lr = 0.001

**警告：如果尝试修改参数，将会是一件非常痛苦的事情，因为我的代码高耦合**

# Why you use MXNET
# 为什么是MXNET？

**Fast!**

**快！**不仅是开发快，运行快，训练也快。

> *当你使用MXNET的时候你会有一种闪电侠在中城奔跑的错觉。*

>                                           *-杜存宵*


# Other versions?
## 我只会用该死的TensorFlow,怎么办？

[Theano](https://github.com/MarkWuNLP/MultiTurnResponseSelection)

你可以选择学习theano或者mxnet，也可以自己实现一个。

*或者自杀*。

# Your code is not pythonic!
## 你的代码就像shit一样

Sorry to hear that.

你可以贡献你pythonic的代码，但是不好意思，我只用了三天的边角零头来开发这个模型。

# Can I use your code to do a chatbot
## 我能用这个东西做个聊天机器人吗

If you can use lucene then you can.

需要为每个query使用lucene检索出一个候选列表，之后进行排序

# How to tipping
## 你真帅，我要为你生猴子

You can sent your money to this [website](https://love.alipay.com/donate/index.htm)

你可以在支付宝E公益进行打赏。

## 你的屁话真多

不好意思，不好意思。




