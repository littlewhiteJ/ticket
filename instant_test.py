from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import json
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv2_drop = nn.Dropout(p = 0.5)
        self.fc1 = nn.Linear(2704, 240)
        self.fc2 = nn.Linear(240, 36)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 2704)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p = 0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class TICKET(torch.utils.data.Dataset):
    def __init__(self, trainpath, testpath, trainlen, testlen, ifTrain):
        super(TICKET, self).__init__()
        self.trainpath = trainpath
        self.testpath = testpath
        self.trainlen = trainlen
        self.testlen = testlen
        self.ifTrain = ifTrain
    def __getitem__(self, index):
        if self.ifTrain == 1:
            path = self.trainpath
        else:
            path = self.testpath
        picpath = path + str(index) + '.png'
        labelpath = path + str(index) + '.txt'
        # print(picpath)
        img = cv2.imread(picpath)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        with open(labelpath, 'r') as f:
            label = f.read(1)
        if label > '9':
            label = ord(label) - ord('A') + 10
        else:
            label = int(label)
        return torch.unsqueeze(torch.tensor(img).type('torch.FloatTensor'), 0), torch.tensor(label)
    def __len__(self):
        if self.ifTrain == 1:
            return self.trainlen
        else:
            return self.testlen

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            print(data)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def load_model(modelpath = 'model/model_all21.dat', device = torch.device("cpu")):
    model = Net().to(device)
    model.load_state_dict(torch.load(modelpath))
    return model

def instant_t(data, model, device = torch.device("cpu")):
    model.eval()
    with torch.no_grad():
        data = torch.tensor([[data]]).to(device).float()
        output = model(data)
        pred = output.max(1, keepdim=True)[1]
        pred = torch.Tensor.numpy(pred)
    label = int(pred[0][0])
    if label > 9:
        predchar = label - 10 + ord('A')
        predchar = chr(predchar)
    else:
        predchar = str(label)

    return predchar

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch WIFI')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    print(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    ###############################################################
    trainfile = 'train03/'
    testfile = 'test03/'
    trainsize = 84
    testsize = 16
    modelpath = 'model/model03.dat'
    ###############################################################
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    dataset_train = WIFI(trainfile, testfile, trainsize, testsize, 1)
    dataset_test = WIFI(trainfile, testfile, trainsize, testsize, 0)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        print('epoch = ' + str(epoch))
        if args.epochs < 3:
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        elif args.epochs < 9:
            optimizer = optim.SGD(model.parameters(), lr=args.lr / 2, momentum=args.momentum)
        elif args.epochs < 25:
            optimizer = optim.SGD(model.parameters(), lr=args.lr / 3, momentum=args.momentum)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr / 4, momentum=args.momentum)
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    torch.save(model.state_dict(), modelpath)

def main_():
    modelpath = 'model/model00.dat'
    data = np.random.rand(130)
    device = torch.device("cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load(modelpath))
    ans = instant_t(data, model)
    print(ans)
    
if __name__ == '__main__':
    main_()