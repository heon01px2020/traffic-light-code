#Also need more data analysis stuff for the test set

import torch
from torch.utils.data import DataLoader
from network import MyNet
from dataset import TrafficLightDataset
from loss import my_loss
from helpers import direction_performance

cuda_available = torch.cuda.is_available()

test_file_loc = ''
test_image_directory = ''

MODEL_PATH = 'path-to-trained-model.pth'

dataset = TrafficLightDataset(csv_file = test_file_loc, root_dir = test_image_directory)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

net = MyNet()
checkpoint = torch.load(MODEL_PATH)
net.load_state_dict(checkpoint, strict=False)
if cuda_available:
    net = net.cuda()

loss_fn = my_loss

net.eval()
running_loss = 0
running_test_performance = 0
total = 0
correct = 0
with torch.no_grad():
    for i, data in enumerate(dataloader):
        images = data['image'].type(torch.FloatTensor)
        mode = data['mode']
        points = data['points']
        if cuda_available:
            images = images.cuda()
            mode = data.cuda()
            points = data.cuda()

        pred_classes, pred_direc = net(images)
        _, predicted = torch.max(pred_classes, 1)
        if(predicted == mode): correct += 1
        
        loss, MSE, cross_entropy =  loss_fn(pred_classes, pred_direc, points, mode)
        running_loss += loss
        #running_test_performance += calculate_performance(pred_direc, points)
        total += 1

print("average loss: " + str(running_loss/total))
print("average performace: " + str(running_test_performance/total))
print("accuracy: " + str(correct/total*100) + "%")


