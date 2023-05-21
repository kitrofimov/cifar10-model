# CIFAR-10 model

## Description

Just a bunch of code I used to fine-tune models from `torchvision`. I am not quite sure if every kind of model architecture is working as I expected, so if you find some bugs, I would highly appreciate if you tell me.

## Usage:
Train a model:
```
git clone https://github.com/fahlerile/cifar10-model.git
python train.py -help
```

**_Sample_** code to predict using your model:
```
model = tv.models.efficientnet_b0()
model.load_state_dict(torch.load(PATH_TO_PTH_FILE, map_location=device))
prediction = model(X)
# X.shape = (1, 3, 224, 224)
```

## Arguments:

>`-h, --help`
>show this help message and exit

>`--name`
>name of your model experiment

>`--path`
>a path to save your model's checkpoints and history.csv file

>`--batch_size`
>a batch size for DataLoaders

>`--learning_rate`
>learning rate. if the training is very slow (not time, but results), then increase it. if the model cannot converge (make loss small), try lowering it

>`--weight_decay`
>L2 regularization parameter, high values try to reduce overfitting

>`--model`
>the model architecture that you want to train. you can choose between efficientnet_b0,alexnet, vgg11 and vgg11_bn

>`--dataset_root`
>where to download the dataset

>`--optimizer`
>optimizer to train the network. you can choose between Adam and SGD

>`--epochs`
>number of epochs to train. low value can lead to underfitting, high value can lead to overfitting (however, L2 regularization (weight_decay) is a technique to deal with it, so you can try increasing it in case of overfitting)

>`--test_every`
>per how many epochs should your model be trained (results are being printed to the console only when the model was trained!)

>`--patience`
>early stopping parameter. if test_loss have been increasing for {patience} epochs, the model would stop training

>`--load_checkpoint`
>path to your checkpoint (if you want to continue training your model from it) IMPORTANT: the model parameter should be the same as in the checkpoint