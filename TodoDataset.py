from torch.utils.data import Dataset

class TodoDataset(Dataset):
    def __init__(self, file_path):
        """ init dataset """
        self.samples = []
        self.file_path = file_path
        self._init_dataset()

    def __len__(self):
        """ returns size of dataset """
        return len(self.samples)

    def __getitem__(self, idx):
        """ returns element by index """
        return self.samples[idx]
    
    def _init_dataset(self):
        """ loads file paths and labels """

        with open(self.file_path + "/words.txt", 'r') as input_file:
            for line in input_file:
                line_split = line.strip().split('\t')
                file_name = self.file_path+"/words/"+line_split[1]
                gt_text = line_split[0]
                self.samples.append((gt_text, file_name))
        input_file.close()




if __name__ == '__main__':

    from torch.utils.data import DataLoader

    dataset = TodoDataset('DATASET')
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=2)
    for i, batch in enumerate(dataloader):
        print(i, batch)

