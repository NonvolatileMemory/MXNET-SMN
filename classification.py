from fr_loss import ArcLoss
from mxnet import init
import mxnet as mx
from mxnet.gluon import nn
import numpy as np
from mxnet import nd
from mxnet.gluon import rnn
import pickle
from mxnet import gluon
from mxnet import autograd
"""
classification orgin classificaiton loss, 1v1
"""
train_size = 31427
batch_size = 100
ctx = mx.gpu(0)
padding_len = 200


train = open('train_class.txt','r')


def get_single_data_train(raw_data,batch_size):
    sent_all,class_id = np.zeros((train_size,padding_len)),[]
    t = 0
    for line in raw_data:
        line = line.strip()
        class_label = line.split('\t')[0]
        sent  = line.split('\t')[1]

        def get_sent(sent):
            sent = sent.split()
            sent = sent[0:padding_len]
            sent = [int(k) for k in sent]
            return [0]*(padding_len-len(sent)) + sent
        sent_all[t] = get_sent(sent)
        class_id.append(class_label)
        t=t+1
        if(t%10000==0):
            print(t)
    sent_all = nd.array(sent_all, ctx = mx.cpu())
    class_id = nd.array(class_id, ctx = ctx)
    print("get data")
    train_dataset = gluon.data.ArrayDataset(sent_all,class_id)
    train_data_iter = gluon.data.DataLoader(train_dataset, batch_size,last_batch='discard', shuffle=True)
    return train_data_iter


# for all
def get_single_data_test_all(raw_data,val_size,test_bs):
    sent_all,class_id = np.zeros((val_size,padding_len)),[]
    t = 0

    for line in raw_data:
        line = line.strip()
        label = line.split('\t')[0]
        sent = line.split('\t')[1]
        sent = sent.split()
        sent = sent[0:padding_len]
        sent = [int(k) for k in sent]

        sent_all[t] = [0]*(padding_len-len(sent)) + sent

        class_id.append(int(label)) # when eval, logits-logits =0 when the logits!=0 , so keep the id>0
        t=t+1
        if(t%10000==0):
            print(t)
    sent_all = nd.array(sent_all, ctx = ctx)
    class_id  = nd.array(class_id,  ctx = ctx)
    print("get data")
    test_dataset = gluon.data.ArrayDataset(sent_all,class_id)
    test_data_iter = gluon.data.DataLoader(test_dataset, test_bs, shuffle=False)
    return test_data_iter



class SMN_Last(nn.Block):
    def __init__(self,**kwargs):
        super(SMN_Last,self).__init__(**kwargs)
        with self.name_scope():
            
            self.emb = nn.Embedding(19224+5,100)
            self.gru_1 = rnn.GRU(layout='NTC',hidden_size=100,bidirectional='True')
            self.pool = nn.GlobalMaxPool1D()
            self.W = self.params.get('param_test',shape=(12877,padding_len))


    def forward(self,question,train,label):
        if(train):
            anc = question[:,0:padding_len]
            def compute_ques(question):
                mask = question.clip(0,1)
                mask = mask.reshape(batch_size,-1,1)
                question = self.emb(question)
                question = self.gru_1(question)
                question = mask*question
                question = nd.transpose(question,(0,2,1))
                question = self.pool(question)
                # question = self.pool2(question)
                question = question.reshape(batch_size,-1)
                question = nd.L2Normalization(question,mode='instance')
                id_center = nd.L2Normalization(self.W.data(),mode='instance')
                res = nd.dot(question,id_center.T)
                return res
            anc = compute_ques(anc)


            #res = nd.dot(question, self.W.data().T)
            return anc
        else:

            q1 = question[:,0:padding_len]
            def compute_ques(question):
                mask = question.clip(0,1)
                mask = mask.reshape(-1,padding_len,1)
                question = self.emb(question)
                question = self.gru_1(question)
                question = mask*question
                question = nd.transpose(question,(0,2,1))
                question = self.pool(question)
                # question = self.pool2(question)
                question = question.reshape(-1,200)
                question = nd.L2Normalization(question,mode='instance')
                return question
            q1 = compute_ques(q1)
            return q1



#Train Model
SMN = SMN_Last()
SMN.initialize(ctx=ctx)



train_iter = get_single_data_train(train,batch_size)
max_epoch = 3000

Sloss = ArcLoss(12877,0.5,64,False)
trainer = gluon.Trainer(SMN.collect_params(), 'adam', {'learning_rate': 0.001})

val_post = open('mpost.txt','r')
val_resp = open('mresp.txt','r')
top_k = 1
val_post_size = 1800
random_size = 500 # test sample number
val_resp_size = val_post_size*random_size
val_size = val_post_size


def test(SMN):

    
    for post,post_label in val_post_iter:
        post_encoding = SMN(post,False,"place_holder")
    post_encoding = post_encoding.reshape((val_post_size,1,-1))
    # val_post_size *1* 100
    xcount = 0
    all_count = 0 
    for resp,label in val_resp_iter:
        res = SMN(resp,False,False) # every raw is the predict for the line
        res = res.reshape((3,random_size,-1))
        res = nd.transpose(res,(0,2,1))
        res = nd.batch_dot(post_encoding[xcount*3:(xcount+1)*3,].copyto(mx.cpu()),res.copyto(mx.cpu()))# yunsuanjieguo
        res = res.reshape(3,-1)
        index = nd.topk(res, ret_typ='indices',k=top_k).reshape(-1,).asnumpy().tolist() # val_size*k,1
        index = [label.asnumpy().tolist()[int(indi)] for indi in index]
        xcount = xcount + 1
        zero_matrix = np.array(index)+1
        all_count = all_count + np.sum(zero_matrix==0)
    print(xcount)
    print("count: " + str(all_count))
    print(" percent: " + str(all_count/(val_size*top_k)))

val_post_iter =  get_single_data_test_all(val_post,val_post_size,val_post_size)
val_resp_iter =  get_single_data_test_all(val_resp,val_resp_size,val_resp_size/600)

for epoch in range(max_epoch):
    train_loss = 0.
    count = 0
    for question,label in train_iter:
        question = question.copyto(ctx)
        label = label.copyto(ctx)
        with autograd.record():
            ques = SMN(question,True,label)
            loss = Sloss(ques,label)
            count = count + 1
            loss.backward()
        trainer.step(batch_size)
        if(True):
            print("loss of epoch "+str(epoch) +" batch "+str(count)+": ")
            print(nd.mean(loss).asscalar())
    if(epoch%4==3):
        test(SMN)

    # if(epoch%30==0):
    # test()

    # acc = mx.metric.Accuracy()#(top_k=1000)
    # acc10 = mx.metric.TopKAccuracy(top_k=5000)
    # acc100 = mx.metric.TopKAccuracy(top_k=100)
    # acc1000 = mx.metric.TopKAccuracy(top_k=1000)

    # pos_mask = nd.array(np.arange(10000),ctx=ctx)
    # neg_mask = np.flipud(np.arange(10000))
    # neg_mask = nd.array(neg_mask,ctx=ctx)

    # print(acc.get())
    # print(acc10.get())
