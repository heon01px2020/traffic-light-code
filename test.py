#Also need more data analysis stuff for the test set

import torch
from torch.utils.data import DataLoader
from network import MyNet
from loss import my_loss
from helpers import direction_performance
from dataset_valid import ValidDataset

cuda_available = torch.cuda.is_available()

test_file_loc = '/home/mv01/Desktop/ISEF 2018/new 5-fold files/test_file.csv'
test_image_directory = '/home/mv01/Desktop/ISEF 2018/resized_photos'

MODEL_PATH = '/home/mv01/Desktop/ISEF 2018/graphs and weights/train_cycle_10_epoch_100_weights3'

dataset = ValidDataset(csv_file = test_file_loc, root_dir = test_image_directory)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

net = MyNet()
checkpoint = torch.load(MODEL_PATH)
net.load_state_dict(checkpoint, strict=False)
if cuda_available:
    net = net.cuda()

loss_fn = my_loss

net.eval()
running_loss = 0
running_test_angle = 0
running_test_start = 0
running_test_end = 0
total = 0
correct = 0
tp = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0}
fp = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0}
fn = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0}
precisions = []
recalls = []
with torch.no_grad():
    for i, data in enumerate(dataloader):
        images = data['image'].type(torch.FloatTensor)
        mode = data['mode']
        points = data['points']
        if cuda_available:
            images = images.cuda()
            mode = mode.cuda()
            points = points.cuda()

        pred_classes, pred_direc = net(images)
        _, predicted = torch.max(pred_classes, 1)
        if(predicted == mode): correct += 1
        if (predicted == mode).sum().item() == 0:
            fp[str(predicted.cpu().numpy()[0])] += 1
            fn[str(mode.cpu().numpy()[0])] += 1
                
        if (predicted == mode).sum().item() == 1:
            tp[str(predicted.cpu().numpy()[0])] += 1
        loss, MSE, cross_entropy =  loss_fn(pred_classes, pred_direc, points, mode)
        running_loss += loss
        angle, start, end = direction_performance(pred_direc, points)
        running_test_angle += angle
        running_test_start += start
        running_test_end += end
        total += 1



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

print("average loss: " + str(running_loss/total))

print("average angle: " + str(running_test_angle/total))
print("average_start: " + str(running_test_start/total))
print("average_end: " + str(running_test_end/total))
print("accuracy: " + str(correct/total*100) + "%")
print(precisions)
print(recalls)
