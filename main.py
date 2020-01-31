import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import cv2 as cv
import random
import argparse
from tqdm import tqdm
import editdistance
from TodoDataset import TodoDataset
from utils import LabelConverter, ImagePreprocess
from Model import Model

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

class Params:
    
    test_image = "data/test.jpg"
    dataset_path = "DATASET/"
    image_size = (32, 192)

def train_model(model, loss, optimizer, scheduler, num_epochs, train_dataloader, val_dataloader, converter):
    
    train_loss = []
    val_loss = []
    char_error = []
    word_acc = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
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
                
                # in case of color image permute dimensions (0 - batch, 3 - colors)
                #inputs = inputs.permute(0, 3, 1, 2).float().to(device) 

                # load images and gt_texts
                inputs = inputs.float().to(device)

                # convert text for CTC Loss input (label - encoded texts, lenghts - lenghts of encoded labels)
                labels, lenghts = converter.encode_text(texts)
                labels = labels.to(device)
                
                batch_size = inputs.size(0)

                optimizer.zero_grad()

                # forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    preds_size = torch.LongTensor([preds.size(0)] * batch_size)
                    loss_value = loss(preds, labels, preds_size, lenghts) / batch_size 

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()
                    elif phase == 'val':
                        batch_char_err, batch_char_total, batch_word_ok, batch_word_total = check_accuracy(preds, texts, converter, silent=True)
                        
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

def check_accuracy(preds, gt_texts, converter, silent=True):

    num_char_err = 0
    num_char_total = 0
    num_word_ok = 0
    num_word_total = 0
    
    # permute dimensions to batch * seq_len * charset lenght
    preds = preds.permute(1, 0, 2)
    
    # decode NN output to characters
    recognized = converter.decode_text(preds.data.cpu().numpy())

    if not silent:
        print('Ground truth -> Recognized')    
    for i in range(len(recognized)):
        num_word_ok += 1 if gt_texts[i] == recognized[i] else 0
        num_word_total += 1
        dist = editdistance.eval(recognized[i], gt_texts[i])
        num_char_err += dist
        num_char_total += len(gt_texts[i])
        if not silent:
            print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + gt_texts[i] + '"', '->', '"' + recognized[i] + '"')
    
    return num_char_err, num_char_total, num_word_ok, num_word_total
    

def validate_model(model, loss, val_dataloader, converter):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()   # Set model to evaluate mode
    running_loss = 0.

    # character/word recognition params
    char_err = 0 
    char_total = 0 
    word_ok = 0 
    word_total = 0

        # Iterate over data.
    for inputs, texts in tqdm(val_dataloader):
                
        # in case of color image permute dimensions (0 - batch, 3 - colors)
        #inputs = inputs.permute(0, 3, 1, 2).float().to(device) 

        # load images and gt_texts
        inputs = inputs.float().to(device)

        # convert text for CTC Loss input (label - encoded texts, lenghts - lenghts of encoded labels)
        labels, lenghts = converter.encode_text(texts)
        labels = labels.to(device)
                
        batch_size = inputs.size(0)

        preds = model(inputs)
        preds_size = torch.LongTensor([preds.size(0)] * batch_size)
        loss_value = loss(preds, labels, preds_size, lenghts) / batch_size 

        batch_char_err, batch_char_total, batch_word_ok, batch_word_total = check_accuracy(preds, texts, converter, silent=False)
                        
        char_err += batch_char_err
        char_total += batch_char_total
        word_ok += batch_word_ok
        word_total += batch_word_total

        running_loss += loss_value.item()
                

    val_loss = running_loss / len(val_dataloader)

    char_error_rate = char_err / char_total
    word_accuracy = word_ok / word_total

    print('Character error rate: %f%%. Word accuracy: %f%%.' % (char_error_rate*100.0, word_accuracy*100.0))


    return val_loss, char_error_rate, word_accuracy

def recognize(model, image, converter):
    " recognize input image to text "
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    image = torch.Tensor(image).to(device)
    image = image.unsqueeze(0) #add batch dimension
    
    pred = model(image)

    # permute dimensions to batch * seq_len * charset lenght
    pred = pred.permute(1, 0, 2)
    
    # decode NN output to characters
    recognized = converter.decode_text(pred.data.cpu().numpy())

    return recognized

def main():
    """ main function """
    # define some command line arguments
    parser = argparse.ArgumentParser(description = 'Todo Bicig handwritten text recognition')
    parser.add_argument('--train', action='store_true', help='train the NN')
    parser.add_argument('--validate', action='store_true', help='validate the NN')

    args = parser.parse_args()

    dataset = TodoDataset(Params.dataset_path, Params.image_size)
    converter = LabelConverter(dataset.char_set)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.train or args.validate:

        #train_loss = [] 
        #val_loss = [] 
        #char_error = [] 
        #word_acc = []
        
        if args.train:

            # split on train and validation sets
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size

            train_set, val_set = random_split(dataset, [train_size, val_size])
            train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=10)
            val_dataloader = DataLoader(val_set, batch_size=32, shuffle=True, num_workers=10)

            # training model
            model = Model(1024, len(dataset.char_set) + 1)
            loss = torch.nn.CTCLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4, amsgrad=True)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

            #train_loss, val_loss, char_error, word_acc = 
            train_model(model=model, loss=loss, optimizer=optimizer, scheduler=scheduler, num_epochs=70, 
                train_dataloader=train_dataloader, val_dataloader=val_dataloader, converter=converter)

            torch.save(model.state_dict(), 'model/model.pth')
        
        if args.validate:
            
            # use all dataset
            val_dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=10)
            loss = torch.nn.CTCLoss()
            
            model = Model(1024, len(dataset.char_set) + 1)
            model.load_state_dict(torch.load('model/model.pth', map_location=device))

            #val_loss, char_error_rate, word_accuracy = 
            validate_model(model, loss, val_dataloader, converter)


    else:
        model = Model(1024, len(dataset.char_set) + 1)
        model.load_state_dict(torch.load('model/model.pth', map_location=device))

        image = ImagePreprocess().resize_image(cv.imread(Params.test_image, cv.IMREAD_GRAYSCALE), Params.image_size)
        result = recognize(model, image, converter)

        print(result)


if __name__ == '__main__':
    main()