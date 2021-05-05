"""
Created on: 2021-02-06
Author: duytinvo
"""
import torch
from utils.core_nns import RNNModel
from utils.data_utils import Txtfile, Data2tensor
from utils.data_utils import SOS, EOS, UNK
from utils.data_utils import SaveloadHP

Data2tensor.set_randseed(1234)

class LMInference:
    def __init__(self, arg_file="./results/lm.args", model_file="./results/lm.m"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args, self.model = self.load_model(arg_file, model_file)

    def load_model(self, arg_file="./results/lm.args", model_file="./results/lm.m"):
        """
        Inputs:
            arg_file: the argument file (*.args)
            model_file: the pretrained model file
        Outputs:
            args: argument dict
            model: a pytorch model instance
        """
        args, model = None, None
        #######################
        # YOUR CODE STARTS HERE
        #load the files 
        args=SaveloadHP.load(arg_file)
        
        model=RNNModel(args.model,len(args.vocab.w2i),args.emsize,args.nhid,
                              args.nlayers,args.dropout,args.tied,args.bidirect)
       
        model.load_state_dict(torch.load(args.trained_model))
        # YOUR CODE ENDS HERE
        #######################
        return args, model

    def generate(self, max_len=1000):
        """
        Inputs:
            max_len: max length of a generated document
        Outputs:
             the text form of a generated document
        """
        doc = [SOS]
        #######################
        # YOUR CODE STARTS HERE
        #get the data from args.vocab.i2w
        data=self.args.vocab.i2w
        #print(len(data))
        #empty list to store label index position of words
        label=[]
        #loop to generate words given a starting word
        for i in range(0,max_len):
            if i==0:
                #when i==0 a random tensor is generated 
                #seed is used so every time a new tensor is generated for i==0 only
                #the tensor generated randomly is the index position of the word
                batch_size = 1
                torch.seed()
                idx=torch.randint(4,len(data), (1,))
                label.append(idx)
                hidden = self.model.init_hidden(batch_size)
            else:
                pass
            #if i==0 then only generate first word once first word is generated 
            #the output of inference function gives a label which is the index
            #postion of the next word then the output of label is passed in forward 
            #and then to inference and we get another word
            #the process is repeated untli max_len or </s> token is encountered
            output, hidden=self.model.forward(idx.reshape(1,-1),hidden)
            #print(hidden)
            p,l=self.model.inference(output)
            if data[l.item()]=='</s>' or len(label)==max_len:
                break
            label.append(l)
            idx=l
        
        #print(len(label))
        #the below list comprehension
        #is used to get words based on the index position in label 
        #it matches the values stored in label which are index position
        #so the values in label are matched wiht values in data 
        #the word residing at that index position is returned and appended to doc list
        doc+=[data[j.item()] for j in label]
        # YOUR CODE ENDS HERE
        #######################
        doc += [EOS]
        return " ".join(doc)

    def recommend(self, context="", topk=5):
        """
        Inputs:
            context: the text form of given context
            topk: number of recommended tokens
        Outputs:
            A list form of recommended words and their probabilities
                e,g, [('i', 0.044447630643844604),
                     ('it', 0.027285737916827202),
                     ("don't", 0.026111900806427002),
                     ('will', 0.023868300020694733),
                     ('had', 0.02248169668018818)]
        """
        rec_wds, rec_probs = [], []
        #######################
        # YOUR CODE STARTS HERE
        #get the data from args.vocab.i2w 
        data=self.args.vocab.i2w
        #get the context data for which the topk values need to be fetched
        context=context.split()
        #split the context into tokens
        #get the index of the words in the context
        idx=[]
        for i in context:
            #print(i)
            for j in range(0,len(data)):
                #if word is present in data then append its index position in idx list
                if i==data[j]:
                    idx.append(j)
                
        #print("index",idx)
        #convert the index to the tensor
        idx=Data2tensor.idx2tensor(idx)
        #print(idx)
        prob=0
        label=0
        batch_size = 1
        hidden = self.model.init_hidden(batch_size)
        #print(hidden)
        output, hidden=self.model.forward(idx.reshape(1,-1),hidden)
        #print(hidden)
        #get the topk words and their probablities
        p,l=self.model.inference(output,topk)
        prob=list(p[0][-1])
        label=list(l[0][-1])
        #print(prob,label)
        #the below list comprehension
        #is used to get words based on the index position in label 
        #it matches the values stored in label which are index position
        #so the values in label are matched wiht values in data 
        #the word residing at that index position is returned and appended to rec_wds list
        rec_wds+=[data[k.item()] for k in label]
        #the prob list contains tensor so item is used to get a number and not a tensor
        #so in below list comprehension p.item() or tensor.item() returns number and not a tensor 
        rec_probs+=[k.item() for k in prob]
        # YOUR CODE ENDS HERE
        #######################
        return list(zip(rec_wds, rec_probs))


if __name__ == '__main__':
    arg_file = "./results/lm.args"
    model_file = "./results/lm.m"
    lm_inference = LMInference(arg_file, model_file)

    max_len = 20
    doc = lm_inference.generate(max_len=max_len)
    print("Random doc: {}".format(doc))
    context = "i went to school"
    topk = 5
    rec_toks = lm_inference.recommend(context=context,topk=topk)
    print("Recommended words of {} is:".format(context))
    for wd, prob in rec_toks:
        print("\t- {} (p={})".format(wd, prob))
    pass