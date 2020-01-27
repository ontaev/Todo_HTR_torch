from torch.utils.data import DataLoader, random_split
import torch
from TodoDataset import TodoDataset
from utils import LabelConverter
from Model import Model
from tqdm import tqdm
import editdistance


def train_model(model, loss, optimizer, scheduler, num_epochs, train_dataloader, val_dataloader, converter):
    
    train_loss = []
    val_loss = []
    char_error = []
    word_acc = []

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

            # character/word recognition params
            char_err = 0 
            char_total = 0 
            word_ok = 0 
            word_total = 0

            # Iterate over data.
            for inputs, texts in tqdm(dataloader):
                
                #inputs = inputs.permute(0, 3, 1, 2).float().to(device)
                inputs = inputs.float().to(device)
                #print(inputs.dtype)
                labels, lenghts = converter.encode_text(texts)
                labels = labels.to(device)
                batch_size = inputs.size(0)

                optimizer.zero_grad()

                # forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    preds_size = torch.Tensor([preds.size(0)] * batch_size)
                    loss_value = loss(preds, labels, preds_size, lenghts) / batch_size 

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()
                    elif phase == 'val':
                        batch_char_err, batch_char_total, batch_word_ok, batch_word_total = check_accuracy(preds, texts, converter)
                        
                        char_err += batch_char_err
                        char_total += batch_char_total
                        word_ok += batch_word_ok
                        word_total += batch_word_total
                # statistics
                running_loss += loss_value.item()
                

            epoch_loss = running_loss / len(dataloader)
            
            print('{} Loss: {:.4f}'.format(phase, epoch_loss), flush=True)

            if phase == 'train':
                train_loss.append(epoch_loss) 
            else:
                val_loss.append(epoch_loss)

                char_error_rate = char_err / char_total
                word_accuracy = word_ok / word_total
                char_error.append(char_error_rate)
                word_acc.append(word_accuracy)
                print('Character error rate: %f%%. Word accuracy: %f%%.' % (char_error_rate*100.0, word_accuracy*100.0))

    return train_loss, val_loss, char_error, word_acc

def check_accuracy(preds, gt_texts, converter):

    num_char_err = 0
    num_char_total = 0
    num_word_ok = 0
    num_word_total = 0

    preds = preds.permute(1, 0, 2)
    print("Preds: ", preds)
    recognized = converter.decode_text(preds.data.cpu().numpy())

    print('Ground truth -> Recognized')    
    for i in range(len(recognized)):
        num_word_ok += 1 if gt_texts[i] == recognized[i] else 0
        num_word_total += 1
        dist = editdistance.eval(recognized[i], gt_texts[i])
        num_char_err += dist
        num_char_total += len(gt_texts[i])
        print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + gt_texts[i] + '"', '->', '"' + recognized[i] + '"')
    
    return num_char_err, num_char_total, num_word_ok, num_word_total
    


def test_model(loader):

    #char_error_rate = num_char_err / num_char_total
    #word_accuracy = num_word_ok / num_word_total
    #print('Character error rate: %f%%. Word accuracy: %f%%.' % (char_error_rate*100.0, word_accuracy*100.0))
    #return char_error_rate

    pass

def main():
    """ main function """
    

    dataset = TodoDataset('DATASET', (32, 192))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    #val_size = len(dataset) - train_size

    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=10)
    val_dataloader = DataLoader(val_set, batch_size=128, shuffle=True, num_workers=10)

    model = Model(256, len(dataset.char_set) + 1)
    loss = torch.nn.CTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-5, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    converter = LabelConverter(dataset.char_set)

    train_model(model=model, loss=loss, optimizer=optimizer, scheduler=scheduler, num_epochs=100, 
                train_dataloader=train_dataloader, val_dataloader=val_dataloader, converter=converter)

    #print(model)
    
    #for i, batch in enumerate(dataloader):
        #print(i, batch)
    #_, texts = next(iter(train_dataloader))
    #print(len(dataset.char_set))
    #print(texts)
    #converter = LabelConverter(dataset.char_set)
    #print(converter.encode_text(texts))

if __name__ == '__main__':
    main()