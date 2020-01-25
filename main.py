from torch.utils.data import DataLoader, random_split
import torch
from TodoDataset import TodoDataset
from utils import LabelConverter
from Model import Model
from tqdm import tqdm


def train_model(model, loss, optimizer, scheduler, num_epochs, train_dataloader, val_dataloader, converter):
    
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                dataloader = val_dataloader
                model.eval()   # Set model to evaluate mode

            running_loss = 0.
            #running_acc = 0.

            # Iterate over data.
            for inputs, texts in tqdm(dataloader):
                
                inputs = inputs.permute(0, 3, 1, 2).float().to(device)
                
                #print(inputs.dtype)
                labels, lenghts = converter.encode_text(texts)

                optimizer.zero_grad()

                # forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    preds_size = torch.Tensor([preds.size(0)] * dataloader.batch_size)
                    loss_value = loss(preds, labels, preds_size, lenghts)
                    #preds_class = preds.argmax(dim=1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                # statistics
                running_loss += loss_value.item()
                #running_acc += (preds_class == labels.data).float().mean()

            epoch_loss = running_loss / len(dataloader)
            #epoch_acc = running_acc / len(dataloader)
            
            if phase == 'train':
                train_loss.append(epoch_loss) 
                #train_acc.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                #val_acc.append(epoch_acc)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss), flush=True)

    return train_loss, train_acc, val_loss, val_acc


def test_model():
    pass

def main():
    """ main function """
    

    dataset = TodoDataset('DATASET', (32, 192))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_set, batch_size=200, shuffle=True, num_workers=100)
    val_dataloader = DataLoader(val_set, batch_size=200, shuffle=True, num_workers=100)

    model = Model(256, len(dataset.char_set))
    loss = torch.nn.CTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=7.0e-4, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    converter = LabelConverter(dataset.char_set)

    train_model(model, loss, optimizer, scheduler, 2, train_dataloader, val_dataloader, converter)

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