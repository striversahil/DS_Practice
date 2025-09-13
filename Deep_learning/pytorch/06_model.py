import torch 
import os
import torch.nn as nn
from torch.utils.data import DataLoader , Dataset
from tqdm import tqdm


class Model(nn.Module):
    def __init__(self, model , model_name , model_path , loss_fn , device , batch_size , learning_rate , start_from_checkpoint = False):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.model_path = model_path
        self.optimizer = None
        self.loss_fn = loss_fn
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.start_from_checkpoint = start_from_checkpoint

        # save path is the path where the model will be saved after training
        self.save_path = os.path.join(self.model_path , self.model_name + ".pth")
        

        self.training_loss = []
        self.validation_loss = []
        self.training_accuracy = []
        self.validation_accuracy = []

        # create model directory if it does not exist
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)

        # load checkpoint if start_from_checkpoint is True. raise error if model already exists and start_from_checkpoint is False
        if self.start_from_checkpoint:
            self.load_checkpoint(self.model_path)
        else:
            if os.path.isfile(self.save_path):
                raise ValueError(f"Model already exists at {self.save_path}. To continue training, set start_from_checkpoint to True.")
            else:
                print(f"Training new model. Model will be saved to {self.save_path}")

        self.setoptimizer()


    def setoptimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters() , lr = self.learning_rate)
        self.optimizer = optimizer

    # load model checkpoint from the specified path
    def load_checkpoint(self , checkpoint_path):
        if os.path.isfile(checkpoint_path):
            check_point = torch.load(checkpoint_path)

            self.model.load_state_dict(check_point['model_state_dict'])
            self.optimizer.load_state_dict(check_point['optimizer_state_dict'])
            self.training_loss = check_point['training_loss']
            self.validation_loss = check_point['validation_loss']

        else:
            raise ValueError(f"No checkpoint found at {checkpoint_path}")

    # save model checkpoint to the specified path
    def save_checkpoint(self , checkpoint_file):
        if not os.path.isdir(checkpoint_file):
            os.makedirs(checkpoint_file)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_loss': self.training_loss,
            'validation_loss': self.validation_loss
        }, checkpoint_file)

        print(f"Model saved to {checkpoint_file}")

    # set dataset for training , validation and testing
    def set_dataset(self , train_set , val_set , test_set):
        
        print("Number of training samples: " , len(train_set))
        print("Number of validation samples: " , len(val_set))
        print("Number of test samples: " , len(test_set))

        self.train_dataloader = DataLoader(train_set , batch_size = self.batch_size , shuffle = True)
        self.val_dataloader = DataLoader(val_set , batch_size = self.batch_size , shuffle = False)
        self.test_dataloader = DataLoader(test_set , batch_size = self.batch_size , shuffle = False)


    # train model for one epoch
    def train_model(self):
        # set model to training mode
        self.model.train()

        for i , (data , label) in enumerate(tqdm(self.train_dataloader , desc = "Training" , leave=False)):

            # move data to device
            data = data.to(self.device)
            label = label.to(self.device)
            
            # forward pass , backward pass and optimization
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.loss_fn(output , label)
            loss.backward()
            self.optimizer.step()

            self.training_loss.append(loss.item())

    # evaluate model on validation set on one epoch
    def evaluate_model(self , train_test_val="val"):
        self.model.eval()
        acc = 0

        data_loader = None
        if train_test_val == "train":
            data_loader = self.train_dataloader
        elif train_test_val == "test":
            data_loader = self.test_dataloader
        elif train_test_val == "val":
            data_loader = self.val_dataloader
        else:
            ValueError("train_test_val must be one of 'train', 'test' or 'val'")

        with torch.no_grad():
            for data , label in tqdm(data_loader , desc = "Evaluating" , leave=False):
                data = data.to(self.device)
                label = label.to(self.device)

                output = self.forward(data)
                loss = self.loss_fn(output , label)

                acc += (output == label).sum().item()

            # calculate accuracy and append to list of accuracies
            if train_test_val == "train":
                self.validation_loss.append(loss.item())
                self.training_accuracy.append(acc / len(self.train_dataloader.dataset))
            elif train_test_val == "test":
                self.validation_loss.append(loss.item())
                self.validation_accuracy.append(acc / len(self.test_dataloader.dataset))
            elif train_test_val == "val":
                self.validation_loss.append(loss.item())
                self.validation_accuracy.append(acc / len(self.val_dataloader.dataset))
            

    def forward(self , x):
        return self.model(x)