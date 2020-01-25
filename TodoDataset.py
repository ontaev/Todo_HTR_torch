from torch.utils.data import Dataset
import cv2 as cv
from utils import LabelConverter



class TodoDataset(Dataset):
    def __init__(self, file_path, image_size):
        """ init dataset """
        self.image_size = image_size
        self.samples = []
        self.file_path = file_path
        self.char_set = ''
        self._init_dataset()
        

    def __len__(self):
        """ returns size of dataset """
        return len(self.samples)

    def __getitem__(self, idx):
        """ returns element by index """
        image = cv.resize(cv.imread(self.samples[idx][0]), self.image_size)
        image = cv.transpose(image)
        gt_text = self.samples[idx][1]
        return image, gt_text
    
    def _init_dataset(self):
        """ loads file paths and labels """
        chars = set()
        with open(self.file_path + "/words.txt", 'r') as input_file:
            for line in input_file:
                line_split = line.strip().split('\t')
                file_name = self.file_path+"/words/"+line_split[1]
                gt_text = line_split[0]
                chars = chars.union(set(list(gt_text)))
                self.samples.append((file_name, gt_text))
        input_file.close()

        self.char_set = sorted(list(chars))


    