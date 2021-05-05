import torch
from utils.core_nns import UniLSTMModel, BiLSTMModel
from utils.data_utils import Txtfile, Data2tensor
from utils.data_utils import SOS, EOS, UNK
from utils.data_utils import SaveloadHP


class SentiInference:
    def __init__(self, arg_file="./results/senti_cls_unilstm.args", model_file="./results/senti_cls_unilstm.m"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args, self.model = self.load_model(arg_file, model_file)

    def load_model(self, arg_file="./results/senti_cls_unilstm.args", model_file="./results/senti_cls_unilstm.m"):
        """
        Inputs:
            arg_file: the argument file (*.args)
            model_file: the pretrained model file
        Outputs:
            args: argument dict
            model: a pytorch model instance
        """
        args = SaveloadHP.load(arg_file)
        self.device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

        ntokens = len(args.vocab.w2i)
        nlabels = len(args.vocab.l2i)
        self.word2idx = args.vocab.wd2idx(args.vocab.w2i, allow_unk=args.allow_unk, start_end=args.se_words)

        if args.bidirect:
            model = BiLSTMModel(args.model, ntokens, args.emb_size, args.hidden_size, args.nlayers, args.dropout,
                                args.bidirect, nlabels=nlabels).to(self.device)
        else:
            model = UniLSTMModel(args.model, ntokens, args.emb_size, args.hidden_size, args.nlayers, args.dropout,
                                      args.bidirect, nlabels=nlabels).to(self.device)
         
        model.load_state_dict(torch.load(model_file))
        model.to(self.device)
        return args, model

    def predict(self, doc="", topk=5):
        """
        Inputs:
            doc: a document
            topk: number of recommended tokens
        Outputs:
            A list form of predicted labels and their probabilities
                e,g, [('5_star', 0.2020701915025711),
                     ('3_star', 0.2010505348443985),
                     ('2_star', 0.2006799429655075),
                     ('1_star', 0.1990940123796463),
                     ('4_star', 0.1971053034067154)]
        """
        doc_ids = self.word2idx(doc.split())
        #######################
        # YOUR CODE STARTS HERE
        pred_lb, pred_probs = None, None
        #convert to tensor
        doc_tensor = Data2tensor.idx2tensor(doc_ids)
        doc_lengths_tensor = Data2tensor.idx2tensor(len(doc_tensor))
        #call the model
        output,_,_=self.model(doc_tensor.unsqueeze(0),doc_lengths_tensor.unsqueeze(0))
        #get the probablities and predicted label
        pred_probs,pred_lb=self.model.inference(output,topk)
        #applied and to list to get individual element out of tensor 
        #as a tensor value cannot be compared with the index positon of labels in args.vocab.i2l
        pred_probs=pred_probs.flatten().tolist()
        pred_lb=pred_lb.flatten().tolist()
        #get label information for the predicted output
        pred_lb=[self.args.vocab.i2l[x] for x in pred_lb]
        # YOUR CODE ENDS HERE
        #######################
        return list(zip(pred_lb, pred_probs))


if __name__ == '__main__':
    #change path information here from unilstm to bilstm or any file generated which needs to be tested
    arg_file = "./results/senti_cls_bilstm.args"
    model_file = "./results/senti_cls_bilstm.m"
    lm_inference = SentiInference(arg_file, model_file)

    doc = "i went to school"
    topk = 5
    pred = lm_inference.predict(doc=doc, topk=topk)
    print("Sentiment prediction of \"{}\" is:".format(doc))
    for lb, prob in pred:
        print("\t- {} (p={})".format(lb, prob))
    pass

