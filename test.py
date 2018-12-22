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
    for i, (images, labels) in enumerate(dataloader):
        if cuda_available:
            images = images.cuda()
            labels = labels.cuda()

        pred_classes, pred_direc = net(images)
        _, predicted = torch.max(pred_classes, 1)
        if(predicted == labels['mode']): correct += 1
        
        running_loss += loss_fn(pred_classes, pred_direc, labels)
        running_test_performance += calculate_performance(pred_direc, labels['points'])
        total += 1

print("average loss: " + running_loss/total)
print("average performace: " + running_test_performance/total)
print("accuracy: " + correct/total*100 + "%")


