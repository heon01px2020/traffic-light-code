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
from dataset_valid import ValidDataset
from loss import my_loss
from helpers import direction_performance

cuda_available = torch.cuda.is_available()

BATCH_SIZE = 32
MAX_EPOCHS = 300
INIT_LR = 0.001
WEIGHT_DECAY = 0.0005
LR_DROP_MILESTONES = [200]
train_file_root = '/home/mv01/Desktop/ISEF 2018/5-fold files/train_file_'
valid_file_root = '/home/mv01/Desktop/ISEF 2018/5-fold files/valid_file_'
image_directory = '/home/mv01/Desktop/ISEF 2018/resized photos'
MODEL_SAVE_PATH = '/home/mv01/Desktop/ISEF 2018/train_cycle_9'

#these save the data for each of the 10 folds
fold_valid_accuracies = []
fold_valid_losses = []
fold_valid_angle = []
fold_valid_start = []
fold_valid_end = []

#10-fold cross validation
for i in range(5):
    
    train_file_loc = train_file_root + str(i+1) + '.csv'
    train_dataset = TrafficLightDataset(csv_file = train_file_loc, root_dir = image_directory)
    valid_file_loc = valid_file_root + str(i+1) + '.csv'
    valid_dataset = ValidDataset(csv_file = valid_file_loc, root_dir = image_directory)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    net = MyNet()
    if cuda_available:
        net = net.cuda()
    
    net.load_state_dict(torch.load('/home/mv01/Desktop/ISEF 2018/graphs and weights/train_cycle_8_epoch_50_weights2'))
    
    loss_fn = my_loss
    
    optimizer = torch.optim.SGD(net.parameters(), lr = INIT_LR, momentum = 0.9, weight_decay = WEIGHT_DECAY)
    #optimizer = torch.optim.Adam(net.parameters(), lr = INIT_LR, weight_decay = WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, LR_DROP_MILESTONES)
    
    #for graphing
    train_losses = []
    train_losses_MSE = []
    train_losses_cross_entropy = []
    valid_losses = []
    valid_losses_MSE  =[]
    valid_losses_cross_entropy = []
    train_accuracies = []
    valid_accuracies = []
    valid_precisions = []
    valid_recalls = []
    val_angles = []
    val_start = []
    val_end = []
    
    
    for epoch in range(MAX_EPOCHS):
       
        ############# 
        #TRAINING
        ############# 
        
        net.train()
        
        running_loss = 0.0 #stores the running loss for 300 photos
        running_loss_MSE = 0.0
        running_loss_cross_entropy = 0.0
        running_loss_epoch = 0.0 #stores the running loss for the entire epoch
        performance_angle = 0.0
        performance_start = 0.0
        performance_end = 0.0
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
            #angle, start, end = direction_performance(pred_direc, points)
            #performance_angle += angle
            #performance_end += end
            #performance_start += start
            loss.backward()
            optimizer.step()
            
            running_loss += loss
            running_loss_MSE += MSE
            running_loss_cross_entropy += cross_entropy
            

            if j % 72 == 71:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, (j + 1)*32, running_loss /81))
                print('mse: ' + str(running_loss_MSE/(j+1)))
                print('cross_entropy: ' + str(running_loss_cross_entropy/(j+1)))
                print("accuracy: " + str(train_correct/train_total/32))
                #print('average performance:' + str(running_performance/300))
                running_loss_epoch += running_loss
                running_loss = 0
                performance_angle = 0.0
                performance_end = 0.0
                performance_start = 0.0
        
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
        
        #for statistics
        tp = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0}
        fp = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0}
        fn = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0}
        precisions = []
        recalls = []
        
        net.eval()
        
        val_running_loss = 0
        val_mse_loss = 0
        val_ce_loss = 0
        val_performance_angle = 0
        val_performance_start = 0
        val_performance_end = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for k, data in enumerate(valid_dataloader, 0):
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
                
                if (predicted == mode).sum().item() == 0:
                    fp[str(predicted.cpu().numpy()[0])] += 1
                    fn[str(mode.cpu().numpy()[0])] += 1
                
                if (predicted == mode).sum().item() == 1:
                    tp[str(predicted.cpu().numpy()[0])] += 1
                
                loss, MSE, cross_entropy = loss_fn(pred_classes, pred_direc, points, mode)
                val_running_loss += loss
                angle, start, end= direction_performance(pred_direc, points)
                val_performance_angle += angle
                val_performance_start += start
                val_performance_end += end
                val_mse_loss += MSE
                val_ce_loss += cross_entropy
            
            #all the statistics
            try:red_precision = tp['0']/(tp['0'] + fp['0'])
            except: red_precision = 0
            precisions.append(red_precision)
            try: red_recall = tp['0']/(tp['0'] + fn['0'])
            except: red_recall = 0
            recalls.append(red_recall)
            
            try: green_precision = tp['1']/(tp['1'] + fp['1'])
            except: green_precision = 0
            precisions.append(green_precision)
            try: green_recall = tp['1']/(tp['1'] + fn['1'])
            except: green_recall = 0
            recalls.append(green_recall)
            
            try: countdown_green_precision = tp['2']/(tp['2'] + fp['2'])
            except: countdown_green_precision = 0
            precisions.append(countdown_green_precision)
            try: countdown_green_recall = tp['2']/(tp['2'] + fn['2'])
            except: countdown_green_recall = 0
            recalls.append(countdown_green_recall)
            
            try: countdown_blank_precision = tp['3']/(tp['3'] + fp['3'])
            except: countdown_blank_precision = 0
            precisions.append(countdown_blank_precision)
            try: countdown_blank_recall = tp['3']/(tp['3'] + fn['3'])
            except: countdown_blank_recall = 0
            recalls.append(countdown_blank_recall)
            
            try: blank_precision = tp['4']/(tp['4'] + fp['4']) 
            except: blank_precision = 0
            precisions.append(blank_precision)
            try: blank_recall = tp['4']/(tp['4'] + fn['4'])
            except: blank_recall = 0
            recalls.append(blank_recall)
            
            try: none_precision = tp['5']/(tp['5'] + fp['5']) 
            except: none_precision = 0
            precisions.append(none_precision)
            try: none_recall = tp['5']/(tp['5'] + fn['5'])
            except: none_recall = 0
            recalls.append(none_recall)
            
            print("epoch: " + str(epoch + 1) + " accuracy over " + str(total) + " validation images: " + str(100*correct/total) + "%")
            valid_accuracies.append(100*correct/total)
            print("average validation loss: " + str(val_running_loss/total))
            print("average validation mse loss: " + str(val_mse_loss/total))
            print("average validation cross-entropy loss: " + str(val_ce_loss/total))
            valid_losses.append(val_running_loss/total)
            valid_losses_MSE.append(val_mse_loss/total)
            valid_losses_cross_entropy.append(val_ce_loss/total)
            print(precisions)
            print(recalls)
            print("direction angle: " + str(val_performance_angle/total))
            print("direction start: " + str(val_performance_start/total))
            print("direction end: " + str(val_performance_end/total))
            val_angles.append(val_performance_angle/total)
            val_start.append(val_performance_start/total)
            val_end.append(val_performance_end/total)
            
            if epoch%100 == 99:
                plt.title('train and validation loss')
                plt.plot(train_losses)
                plt.plot(valid_losses)
                plt.show()
            
            
            if epoch == 100:
                torch.save(net.state_dict(), MODEL_SAVE_PATH + '_epoch_100_weights' + str(i+1))

    #Plot graphs of valid and train
    plt.title('train vs validation loss')
    plt.plot(valid_losses)
    plt.plot(train_losses)
    plt.savefig(MODEL_SAVE_PATH + 'losses' + str(i+1))
    plt.show()
    plt.title('train vs validation accuracies')
    plt.plot(valid_accuracies)
    plt.plot(train_accuracies)
    plt.savefig(MODEL_SAVE_PATH + 'accuracies' + str(i+1))
    plt.show()
    plt.title('train and valid cross entropy')
    plt.plot(train_losses_cross_entropy)
    plt.plot(valid_losses_cross_entropy)
    plt.savefig(MODEL_SAVE_PATH + 'train_valid_ce' + str(i+1))
    plt.show()
    plt.title('train and valid MSE')
    plt.plot(train_losses_MSE)
    plt.plot(valid_losses_MSE)
    plt.savefig(MODEL_SAVE_PATH + 'train_valid_MSE' + str(i+1))
    plt.show()

    #save the validation data so we can graph validation of each fold later
    fold_valid_accuracies.append(valid_accuracies)
    fold_valid_angle.append(val_angles)
    fold_valid_start.append(val_start)
    fold_valid_end.append(val_end)
    fold_valid_losses.append(valid_losses)
    
    torch.save(net.state_dict(), MODEL_SAVE_PATH + '_final_weights' + str(i+1)) #save the model weights
    
    
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
