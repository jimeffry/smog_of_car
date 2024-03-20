from __future__ import print_function

import argparse
import os
import os.path

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import codecs
import errno

WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
class MyTrainData(data.Dataset):

    urls = [
        'http://9nadmin.jd.local/upload/lilingyun/processed/train-images-idx3-ubyte.gz',
        'http://9nadmin.jd.local/upload/lilingyun/processed/train-labels-idx1-ubyte.gz',
        'http://9nadmin.jd.local/upload/lilingyun/processed/t10k-images-idx3-ubyte.gz',
        'http://9nadmin.jd.local/upload/lilingyun/processed/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, train=True, transform=None, target_transform=None, download=False):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if download:
           self.download()
        if not self._check_exists():
            raise RuntimeError(' dataset not found')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.training_file))
    
    def __len__(self):
        """
        """
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
    
    def __getitem__(self, index):
        """
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
        
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    
    def _check_exists(self):
        """
        """
        print(os.getcwd())
        return os.path.exists(os.path.join(os.getcwd(), self.training_file)) and \
            os.path.exists(os.path.join(os.getcwd(), self.test_file))
    
    def download(self):
        
        from six.moves import urllib
        import gzip
        if self._check_exists():
            return
        
        try:
            os.makedirs(os.path.join(os.getcwd(), self.raw_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                false
        """ 
        for url in self.urls:
            print('download' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(os.getcwd(), self.test_file)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)
        """
        print('Processing ...')

        training_set = {
            read_image_file(os.path.join(os.getcwd(), 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(os.getcwd(), 'train-labels-idx1-ubyte'))
        }
        test_set = {
            read_image_file(os.path.join(os.getcwd(), 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(os.getcwd(), 't10k-labels-idx1-ubyte'))
        }
        with open(os.path.join(os.getcwd(), self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(os.getcwd(), self.test_file), 'wb') as f:
            torch.save(test_set, f)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

def get_int(b):
    print(int(codecs.encode(b, 'hex'), 16))
    return int(codecs.encode(b, 'hex'), 16)

def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        print(data[:4])
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()

def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        print(data[:4])
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols).long()


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
    
    def forward(self, pred, truth):
        truth = torch.mean(truth, 1)
        truth = torch.view(-1, 2048)
        pred = pred.view(-1, 2048)
        return torch.mean(torch.mean((pred-truth)**2, 1), 0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def train(args, model, device, train_loader, optimizer, epoch, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            niter = epoch * len(train_loader) + batch_idx
            writer.add_scalar('loss', loss.item(), niter)

def test(args, model, device, test_loader, writer, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\naccuracy={:.4f}\n'.format(float(correct) / len(test_loader.dataset)))
    writer.add_scalar('accuracy', float(correct) / len(test_loader.dataset), epoch)


def should_distribute():
    return dist.is_available() and WORLD_SIZE > 1


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
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
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dir', default='models', metavar='L',
                        help='directory where summary logs are stored')
    if dist.is_available():
        parser.add_argument('--backend', type=str, help='Distributed backend',
                            choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
                            default=dist.Backend.GLOO)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print('Using CUDA')

    writer = SummaryWriter(args.dir)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(WORLD_SIZE)
    if should_distribute():
        print('Using distributed PyTorch with {} backend'.format(args.backend))
        dist.init_process_group(backend=args.backend)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    data_tf = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
    train_data = MyTrainData(train=True, transform=data_tf, target_transform=None, download=True)
    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_data = MyTrainData(train=False, transform=data_tf, target_transform=None, download=True)
    test_loader = data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    """
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)
    """

    model = Net().to(device)
    #分发模型
    if is_distributed():
        Distributor = nn.parallel.DistributedDataParallel if use_cuda \
            else nn.parallel.DistributedDataParallelCPU
        model = Distributor(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    #设置epoch数
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, writer)
        test(args, model, device, test_loader, writer, epoch)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()
