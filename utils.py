import torch
import cv2 as cv
import numpy as np

class ImagePreprocess:
    def __init__(self):
        pass

    def resize_image(self, image, image_size):
        # transpose vertical Todo image
        #image = cv.transpose(image)

        # create target image and copy sample image into it
        (wt, ht) = image_size
        (h, w, _) = image.shape
        fx = w / wt
        fy = h / ht
        f = max(fx, fy)
        newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)
        image = cv.resize(image, newSize)
    
        target = np.ones([ht, wt, 3]) * 255
        target[0:newSize[1], 0:newSize[0], :] = image
        image = cv.transpose(target)

      
        return image


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
            label_str = [self.char_set.index(c) + 1 for c in text]
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

    def decode_text(self, encoded_text):
        "extract texts from output of CTC decoder"
        text = []
        for encoded in encoded_text:
            char_list = []
            for i in encoded:
                if i != 0:
                    char_list.append(self.char_set[i - 1])
            text.append(''.join(char_list))
        return text


