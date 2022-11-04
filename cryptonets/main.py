import torch
import torch.optim as optim
from torch.autograd import Variable
import argparse
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import model
import torch.nn.functional as F
import matplotlib.pyplot as plt


# reference: https://github.com/gsp-27/pytorch_Squeezenet

parser = argparse.ArgumentParser('Options for training CryptoNets in pytorch')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size of train')
parser.add_argument('--epoch', type=int, default=20, metavar='N', help='number of epochs to train for')
parser.add_argument('--learning-rate', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False, help='use cuda for training')
parser.add_argument('--log-schedule', type=int, default=100, metavar='N', help='number of epochs to save snapshot after')
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--model_name', type=str, default=None, help='Use a pretrained model')
parser.add_argument('--want_to_test', type=bool, default=False, help='make true if you just want to test')
parser.add_argument('--num_classes', type=int, default=10, help="how many classes training for")
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=False, download=True,
                    transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)


net  = model.CryptoNets()
if args.model_name is not None:
    print("loading pre trained weights")
    pretrained_weights = torch.load(args.model_name)
    net.load_state_dict(pretrained_weights)

if args.cuda:
    net.cuda()

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

print(get_parameter_number(net))

avg_loss = list()
best_accuracy = 0.0
fig1, ax1 = plt.subplots()

optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

def train(epoch):
    global avg_loss
    correct = 0
    net.train()
    for b_idx, (data, targets) in enumerate(train_loader):
        if args.cuda:
            data, targets = data.cuda(), targets.cuda()
        data, targets = Variable(data), Variable(targets)
        optimizer.zero_grad()
        scores = net.forward(data)
        scores = scores.view(len(scores), args.num_classes)
        loss = F.nll_loss(scores, targets)
        pred = scores.data.max(1)[1] 
        correct += pred.eq(targets.data).cpu().sum()
        avg_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        if b_idx % args.log_schedule == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (b_idx+1) * len(data), len(train_loader.dataset),
                100. * (b_idx+1)*len(data) / len(train_loader.dataset), loss.item()))
            ax1.plot(avg_loss)
            fig1.savefig("my_cryptonets_loss.jpg")
    train_accuracy = correct / float(len(train_loader.dataset))
    print("training accuracy ({:.2f}%)".format(100*train_accuracy))
    return (train_accuracy*100.0)


def val():
    global best_accuracy
    correct = 0
    net.eval()
    total_examples = 0
    for idx, (data, target) in enumerate(test_loader):
        if idx == 73:
            break
        total_examples += len(target)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        score = net.forward(data)
        pred = score.data.max(1)[1]
        pred = pred.view(1, len(pred))
        correct += pred.eq(target.data).cpu().sum()

    print("predicted {} out of {}".format(correct, total_examples))
    val_accuracy = correct / float(total_examples) * 100.0
    print("accuracy = {:.2f}%".format(val_accuracy))

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(net.state_dict(),'cryptonets.pth')
    return val_accuracy

def test():
    weights = torch.load('cryptonets.pth')
    net.load_state_dict(weights)
    net.eval()

    test_correct = 0
    total_examples = 0
    for idx, (data, target) in enumerate(test_loader):
        if idx < 73:
            continue
        total_examples += len(target)
        data, target = Variable(data), Variable(target)
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        scores = net(data)
        pred = scores.data.max(1)[1]
        pred = pred.view(1, len(pred))
        test_correct += pred.eq(target.data).cpu().sum()
    print("Predicted {} out of {} correctly".format(test_correct, total_examples))
    return 100.0 * test_correct / (float(total_examples))

if __name__ == '__main__':
    if not args.want_to_test:
        fig2, ax2 = plt.subplots()
        train_acc, val_acc = list(), list()
        for i in range(1,args.epoch+1):
            train_acc.append(train(i))
            val_acc.append(val())
            ax2.plot(train_acc, 'g')
            ax2.plot(val_acc, 'b')
            fig2.savefig('train_val_accuracy.jpg')
    else:
        test_acc = test()
        print("Testing accuracy on MNIST data is {:.2f}%".format(test_acc))
