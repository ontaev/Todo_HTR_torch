import torch

class LabelConverter:
    def __init__(self, char_set, ignore_case=False):
        self.char_set = char_set
        if ignore_case:
            char_set = char_set.lower()
        
    def encode_text(self, texts):
        "puts ground truth texts into tensor for ctc_loss and returns lenghts of labels"

        lenghts = []
        
        indices = []
        values = []
        shape = [len(texts), 0] 

	    # go over all texts
        for (batch_element, text) in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            label_str = [self.char_set.index(c) for c in text]
            # add lenght of text to list 
            lenghts.append(len(label_str))
            # sparse tensor must have size of max. label-string
            if len(label_str) > shape[1]:
                shape[1] = len(label_str)
            # put each label into sparse tensor
            for (i, label) in enumerate(label_str):
                indices.append([batch_element, i])
                values.append(label)
            
        i = torch.LongTensor(indices)
        v = torch.LongTensor(values)
        result = torch.sparse.LongTensor(i.t(), v, shape).to_dense()

        lenghts = torch.Tensor(lenghts)

        return result, lenghts

    def decode_text(self, ctc_output, batch_size):
        "extract texts from output of CTC decoder"
