import torch
import cv2 as cv
import numpy as np
from CTCBestPath import ctcBestPath

class ImagePreprocess:
    def __init__(self):
        pass

    def resize_image(self, image, image_size):
        # transpose vertical Todo image
        #image = cv.transpose(image)

        # create target image and copy sample image into it
        (wt, ht) = image_size
        #(h, w, _) = image.shape - for color image case
        (h, w) = image.shape
        fx = w / wt
        fy = h / ht
        f = max(fx, fy)
        newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)
        image = cv.resize(image, newSize)
    
        target = np.ones([ht, wt]) * 255
        target[0:newSize[1], 0:newSize[0]] = image
        #target = np.ones([ht, wt, 3]) * 255
        #target[0:newSize[1], 0:newSize[0], :] = image
        image = cv.transpose(target)

      
        return image


class LabelConverter:
    def __init__(self, char_set, ignore_case=False):
        self.char_set = char_set
        if ignore_case:
            char_set = char_set.lower()
        
    def encode_text(self, text):
        "puts ground truth texts into tensor for ctc_loss and returns lenghts of labels"

        length = []
        result = []
        for item in text:            
            
            length.append(len(item))
            r = []
            for char in item:
                index = self.char_set.index(char)
                
                r.append(index)
            result.append(r)
        
        max_len = 0
        for r in result:
            if len(r) > max_len:
                max_len = len(r)
        
        result_temp = []
        for r in result:
            for _ in range(max_len - len(r)):
                r.append(0)
            result_temp.append(r)

        text = result_temp
        return (torch.LongTensor(text), torch.LongTensor(length))

    def decode_text(self, encoded_text):
        "extract texts from output of CTC decoder"

        text = []
        
        for encoded in encoded_text:
            text.append(ctcBestPath(encoded, self.char_set))
        return text