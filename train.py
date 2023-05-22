import torchvision as tv
from torchvision import transforms

from wrapper import ClassificationModelWrapper
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(prog='train.py',
                                 description='Train a CIFAR-10 model')
parser.add_argument('--name', type=str, help='name of your model experiment')
parser.add_argument('--path', default=None, type=Path, help="a path to save your model's checkpoints and history.csv file")
parser.add_argument('--batch_size', default=32, type=int, help="a batch size for DataLoaders")
parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate. if the training is very slow (not time, but results),\
                                                                       then increase it. if the model cannot converge (make loss small), try lowering it')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='L2 regularization parameter, high values try to reduce overfitting')
parser.add_argument('--model', default='efficientnet_b0', type=str, help='the model architecture that you want to train. you can choose\
                                                                          between efficientnet_b0, alexnet, vgg11 and vgg11_bn')
parser.add_argument('--dataset_root', default=Path('./CIFAR-10'), type=Path, help='where to download the dataset')
parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer to train the network. you can choose between Adam and SGD')
parser.add_argument('--epochs', default=50, type=int, help='number of epochs to train. low value can lead to underfitting, high value can lead to\
                                                            overfitting (however, L2 regularization (weight_decay) is a technique to deal with it,\
                                                            so you can try increasing it in case of overfitting)')
parser.add_argument('--test_every', default=5, type=int, help='per how many epochs should your model be trained (results are being printed to the console\
                                                               only when the model was trained!)')
parser.add_argument('--patience', default=None, type=int, help='early stopping parameter. if test_loss have been increasing for {patience} epochs,\
                                                                the model would stop training')
parser.add_argument('--load_checkpoint', default=None, type=Path, help='path to your checkpoint (if you want to continue training your model from it)\
                                                                        IMPORTANT: the model parameter should be the same as in the checkpoint')

args = parser.parse_args()

if args.path is None:
    path = Path(f'./{args.name}')
else:
    path = args.path

wrapper = ClassificationModelWrapper(args.batch_size, args.learning_rate, args.weight_decay, args.name, path)

wrapper.load_model(num_classes=10, model=args.model, checkpoint=args.load_checkpoint)

transform = transforms.Compose([
      transforms.Resize((256, 256)),
      transforms.CenterCrop((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
])

train_dataset = tv.datasets.CIFAR10(
    root=args.dataset_root,
    train=True,
    download=True,
    transform=transform
)

test_dataset = tv.datasets.CIFAR10(
    root=args.dataset_root,
    train=False,
    download=True,
    transform=transform
)

wrapper.prepare_dataloaders(train_dataset, test_dataset)
wrapper.init_optim(optimizer=args.optimizer)
history = wrapper.train(args.epochs, args.test_every, args.patience)
history.to_csv(path / 'history.csv')
print(f'Done! Model checkpoints and history.csv file are in {str(Path(path))} directory. \
The names of checkpoints are following the pattern EPOCH_TESTLOSS_TESTACCURACY')
