# CIFAR-10 model

Scripts to fine-tune torchvision.models on CIFAR-10 dataset. Please, submit an issue or pull request if you find a bug!

## Usage:
Train a model:
```bash
git clone https://github.com/kitrofimov/cifar10-model.git
cd cifar10-model
poetry install
poetry run python train.py [ARGS]
```

**_Sample_** code to predict using your model:
```py
model = tv.models.efficientnet_b0()
model.load_state_dict(torch.load(PATH_TO_PTH_FILE, map_location=device))
prediction = model(X)
# X.shape = (1, 3, 224, 224)
```

## Arguments:

```
usage: train.py [-h] [--name NAME] [--path PATH] [--batch_size BATCH_SIZE]
                [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY]
                [--model MODEL] [--dataset_root DATASET_ROOT]
                [--optimizer OPTIMIZER] [--epochs EPOCHS]
                [--test_every TEST_EVERY] [--patience PATIENCE]
                [--load_checkpoint LOAD_CHECKPOINT]

Train a CNN model using CIFAR-10 dataset (https://www.cs.toronto.edu/~kriz/cifar.html)

options:
  -h, --help            show this help message and exit
  --name NAME           name of your model experiment
  --path PATH           a path to save your model's checkpoints and
                        history.csv file
  --batch_size BATCH_SIZE
                        a batch size for DataLoaders
  --learning_rate LEARNING_RATE
                        learning rate. if the training is very slow (not time,
                        but results), then increase it. if the model cannot
                        converge (make loss small), try lowering it
  --weight_decay WEIGHT_DECAY
                        L2 regularization parameter, high values try to reduce
                        overfitting
  --model MODEL         the model architecture that you want to train. you can
                        choose between efficientnet_b0, alexnet, vgg11 and
                        vgg11_bn (https://pytorch.org/vision/stable/models.html)
  --dataset_root DATASET_ROOT
                        where to download the dataset
  --optimizer OPTIMIZER
                        optimizer to train the network. you can choose between
                        Adam and SGD
  --epochs EPOCHS       number of epochs to train. low value can lead to
                        underfitting, high value can lead to overfitting
                        (however, L2 regularization (weight_decay) is a
                        technique to deal with it, so you can try increasing
                        it in case of overfitting)
  --test_every TEST_EVERY
                        per how many epochs should your model be trained
                        (results are being printed to the console only when
                        the model was trained!)
  --patience PATIENCE   early stopping parameter. if test_loss have been
                        increasing for {patience} epochs, the model would stop
                        training
  --load_checkpoint LOAD_CHECKPOINT
                        path to your checkpoint (if you want to continue
                        training your model from it) IMPORTANT: the model
                        parameter should be the same as in the checkpoint
```
