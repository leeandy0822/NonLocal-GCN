import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data,DataLoader
from torch_geometric.nn import GCNConv
from torch_scatter import  scatter_max
import matplotlib.pyplot as plt
from load_mnist_graph import load_mnist_graph
import sys
import wandb
from args import Option

sys.path.append('Non-local_pytorch')

from lib.non_local_embedded_gaussian import NONLocalBlock2D as NONEmbed_gaussian
from lib.non_local_dot_product import NONLocalBlock2D as NONDot_product
from lib.non_local_concatenation import NONLocalBlock2D as NONConcate
from lib.non_local_gaussian import NONLocalBlock2D as NONGaussian



class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.conv1 = GCNConv(2, 16)
        
        # define non-local operator type
        self.normal_gcn = False
        if args.NonLocal_Operator == "embed_gaussian":
            self.nonlocal1 = NONEmbed_gaussian(in_channels=1)
        elif args.NonLocal_Operator == "gaussian":
            self.nonlocal1 = NONGaussian(in_channels=1)
        elif args.NonLocal_Operator == "concate":
            self.nonlocal1 = NONConcate(in_channels=1)
        elif args.NonLocal_Operator == "dot":
            self.nonlocal1 = NONDot_product(in_channels=1)
        else:
            self.normal_gcn = True 
            
        
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 48)
        self.conv4 = GCNConv(48, 64)
        self.conv5 = GCNConv(64, 96)
        self.conv6 = GCNConv(96, 128)
        self.linear1 = torch.nn.Linear(48,24)
        self.linear2 = torch.nn.Linear(24,10)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
    
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        if self.normal_gcn:
            pass
        else:
            x = x.view(self.args.batch,1,16,-1)
            x = self.nonlocal1(x)
            x = x.view(-1,16)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        if self.normal_gcn:
            pass
        else:
            x = x.view(self.args.batch,1,32,-1)
            x = self.nonlocal1(x)
            x = x.view(-1,32)
            
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # x = self.conv4(x, edge_index)
        # x = F.relu(x)
        # x = self.conv5(x, edge_index)
        # x = F.relu(x)
        # x = self.conv6(x, edge_index)
        # x = F.relu(x)
        
        x, _ = scatter_max(x, data.batch, dim=0)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


def main():
    
    args = Option().create()
    
    wandb.init(
    project="NonLocal-GCN", 
    entity= "leeandy0822",
    name = args.NonLocal_Operator + str(args.epochs),
    config={
    "epochs": 15,
    })
    
    data_size = 60000
    train_size = 50000
    batch_size = args.batch
    epoch_num = args.epochs
    
    #前準備
    mnist_list = load_mnist_graph(data_size=data_size)
    device = torch.device('cuda')
    model = Net(args).to(device)
    trainset = mnist_list[:train_size]
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    testset = mnist_list[train_size:]
    testloader = DataLoader(testset, batch_size=batch_size)
    
    criterion = nn.CrossEntropyLoss()
    history = {
        "train_loss": [],
        "test_loss": [],
        "test_acc": []
    }

       
    print("Start Train")
    
    #学習部分
    model.train()
    for epoch in range(epoch_num):
        train_loss = 0.0
        for i, batch in enumerate(trainloader):
            batch = batch.to("cuda")
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs,batch.t)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.cpu().item()
            if i % 10 == 9:
                progress_bar = '['+('='*((i+1)//10))+(' '*((train_size//100-(i+1))//10))+']'
                print('\repoch: {:d} loss: {:.3f}  {}'
                        .format(epoch + 1, loss.cpu().item(), progress_bar), end="  ")


        print('\repoch: {:d} loss: {:.3f}'
            .format(epoch + 1, train_loss / (train_size / batch_size)), end="  ")
        
        history["train_loss"].append(train_loss / (train_size / batch_size))

        correct = 0
        total = 0
        batch_num = 0
        loss = 0
        with torch.no_grad():
            for data in testloader:
                data = data.to(device)
                outputs = model(data)
                loss += criterion(outputs,data.t)
                _, predicted = torch.max(outputs, 1)
                total += data.t.size(0)
                batch_num += 1
                correct += (predicted == data.t).sum().cpu().item()

        history["test_acc"].append(correct/total)
        history["test_loss"].append(loss.cpu().item()/batch_num)
        endstr = ' '*max(1,(train_size//1000-39))+"\n"
        print('Test Accuracy: {:.2f} %%'.format(100 * float(correct/total)), end='  ')
        print(f'Test Loss: {loss.cpu().item()/batch_num:.3f}',end=endstr)
        
        wandb.log({'epoch': epoch, 
                   'Test Accuracy':float(correct/total), 
                   'Test Loss': loss.cpu().item()/batch_num,
                    'Loss' : train_loss / (train_size / batch_size)
                   })


    print('Finished Training')

    #最終結果出力
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += data.t.size(0)
            correct += (predicted == data.t).sum().cpu().item()
    print('Accuracy: {:.2f} %%'.format(100 * float(correct/total)))

if __name__=="__main__":
    main()
