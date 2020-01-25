from torch.utils.data import DataLoader
from TodoDataset import TodoDataset
from utils import LabelConverter
from Model import Model

def train():
    pass

def validate():
    pass

def infer():
    pass

def main():
    """ main function """
    

    dataset = TodoDataset('DATASET', (28, 196))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

    model = Model(256, len(dataset.char_set))

    print(model)
    
    #for i, batch in enumerate(dataloader):
        #print(i, batch)
    #_, texts = next(iter(dataloader))
    #print(dataset.char_set)
    #print(texts)
    #converter = LabelConverter(dataset.char_set)
    #print(converter.to_sparse(texts))

if __name__ == '__main__':
    main()