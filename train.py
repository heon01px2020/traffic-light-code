##################################################################################################
#Things to add:
#More Data Analysis such as recall and precision
#Data Analysis for direction; show the performance for when the zebra crossing is blocked vs unblocked
#IM SURE I FORGOT QUITE A LOT OF OTHER STUFF LMAO
##################################################################################################

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from network import MyNet
from dataset import TrafficLightDataset
from loss import my_loss
#from helpers import direction_performance

cuda_available = torch.cuda.is_available()

BATCH_SIZE = 32
MAX_EPOCHS = 1000
INIT_LR = 0.1
WEIGHT_DECAY = 0.00001
LR_DROP_MILESTONES = [50,150,250,350,450,550,650,750,850,950]
train_file_root = '/home/mv01/Desktop/ISEF 2018/rewritten files/train_file_'
valid_file_root = '/home/mv01/Desktop/ISEF 2018/rewritten files/valid_file_'
image_directory = '/home/mv01/Desktop/ISEF 2018/resized photos'
MODEL_SAVE_PATH = '/home/mv01/Desktop/ISEF 2018'

#these save the data for each of the 10 folds
fold_valid_accuracies = []
fold_valid_losses = []
#fold_valid_performances = []

#10-fold cross validation
for i in range(1):
    
    train_file_loc = train_file_root + str(i+1) + '.csv'
    train_dataset = TrafficLightDataset(csv_file = train_file_loc, root_dir = image_directory)
    valid_file_loc = valid_file_root + str(i+1) + '.csv'
    valid_dataset = TrafficLightDataset(csv_file = valid_file_loc, root_dir = image_directory)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    net = MyNet()
    if cuda_available:
        net = net.cuda()
        
    loss_fn = my_loss
    
    #optimizer = torch.optim.SGD(net.parameters(), lr = INIT_LR, momentum = 0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr = INIT_LR, weight_decay = WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, LR_DROP_MILESTONES)
    
    #for graphing
    train_losses = []
    train_losses_MSE = []
    train_losses_cross_entropy = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    #valid_performances = []
    
    for epoch in range(MAX_EPOCHS):
       
        ############# 
        #TRAINING
        ############# 
        
        net.train()
        
        running_loss = 0.0 #stores the running loss for 300 photos
        running_loss_MSE = 0.0
        running_loss_cross_entropy = 0.0
        running_loss_epoch = 0.0 #stores the running loss for the entire epoch
        #running_performance = 0.0
        train_correct = 0
        train_total = 0
        for j, data in enumerate(train_dataloader, 0): 
            optimizer.zero_grad()
            train_total += 1
            images = data['image'].type(torch.FloatTensor)
            mode = data['mode']
            points = data['points']
            if cuda_available:
                images = images.cuda()
                mode = mode.cuda()
                points = points.cuda()
            
            pred_classes, pred_direc = net(images)
            _, predicted = torch.max(pred_classes, 1)
            train_correct += (predicted == mode).sum().item()
            loss, MSE, cross_entropy = loss_fn(pred_classes, pred_direc, points, mode)
            #performance = direction_performance(pred_direc, points)
            loss.backward()
            optimizer.step()
            
            running_loss += loss
            running_loss_MSE += MSE
            running_loss_cross_entropy += cross_entropy
            #running_performance += performance

            if j % 9 == 8:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, (j + 1)*32, running_loss /9))
                print('mse: ' + str(running_loss_MSE/(j+1)))
                print('cross_entropy: ' + str(running_loss_cross_entropy/(j+1)))
                print('epoch: ' + str(epoch+1) + " accuracy: " + str(train_correct/train_total/32))
                #print('average performance:' + str(running_performance/300))
                running_loss_epoch += running_loss
                running_loss = 0
                #running_performance = 0
        
        train_losses_MSE.append(running_loss_MSE/train_total)
        running_loss_MSE = 0.0
        train_losses_cross_entropy.append(running_loss_cross_entropy/train_total)
        running_loss_cross_entropy = 0.0
        train_losses.append(running_loss_epoch/train_total) #store the epochs running loss
        train_accuracies.append(train_correct/train_total/32*100) #store the accuracy for the epoch
                
        #lr_scheduler.step(epoch + 1)   
        
        
        ############# 
        #VALIDATION
        ############# 
        
        net.eval()
        
        val_running_loss = 0
        #val_performance_total = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, data in enumerate(valid_dataloader, 0):
                images = data['image'].type(torch.FloatTensor)
                mode = data['mode']
                points = data['points']
                if cuda_available:
                    images = images.cuda()
                    mode = mode.cuda()
                    points = points.cuda()
                
                pred_classes, pred_direc = net(images)
                _, predicted = torch.max(pred_classes, 1)
                total += 1
                correct += (predicted == mode).sum().item()
                lots, MSE, cross_entropy = loss_fn(pred_classes, pred_direc, points, mode)
                val_running_loss += loss
                #val_performance_total += direction_performance(pred_direc, points)
            
    
            print("epoch: " + str(epoch+1) +" accuracy over " + str(total) + " validation images: " + str(100*correct/total) + "%")
            valid_accuracies.append(100*correct/total)
            print("average validation loss: " + str(val_running_loss/total))
            valid_losses.append(val_running_loss/total)
            #print("average direction performance: " +str(val_performance_total/total))
            #valid_performances.append(val_performance_total/total)
            if epoch == 300:
                torch.save(net.state_dict(), MODEL_SAVE_PATH + '_epoch_300_weights')
    
    #Plot graphs of valid and train
    plt.title('train vs validation loss')
    plt.plot(valid_losses)
    plt.plot(train_losses)
    plt.show()
    plt.title('difference in mse and cross-entropy')
    plt.plot(train_losses_MSE)
    plt.plot(train_losses_cross_entropy)
    plt.show()
    plt.title('train vs validation accuracies')
    plt.plot(valid_accuracies)
    plt.plot(train_accuracies)
    plt.show()
    #plt.plot(valid_performances)
    #plt.show()
    
    #save the validation data so we can graph validation of each fold later
    fold_valid_accuracies.append(valid_accuracies)
    #fold_valid_performances.append(valid_performances)
    fold_valid_losses.append(valid_losses)
    
    torch.save(net.state_dict(), MODEL_SAVE_PATH + str(i)) #save the model weights
    
    
#graph the data from each fold
#accuracy
for n in fold_valid_accuracies:
    plt.plot(n)
plt.title('fold validation accuracies')
plt.show()
#loss
for n in fold_valid_losses:
    plt.plot(n)
plt.title('fold validation losses')
plt.show()
#performance
#for n in fold_valid_performances:
#    plt.plot(n)
#plt.show()