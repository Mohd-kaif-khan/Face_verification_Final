import torch
import torch.nn as nn
import torch.nn.functional as F

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


class CNN(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5,padding=2)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=2, padding=1)
        self.dropout = nn.Dropout(p=0.20)
        self.linear_shape = None
        x = torch.randn(3,200,200).view(-1,3,200,200)
        x = self.convs(x)
        self.linear1 = nn.Linear(self.linear_shape, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.output = nn.Linear(512,256)

    def convs(self,x):
        # print("Initial shape :- ",x.shape)
        x = F.max_pool2d(F.leaky_relu(self.conv1(x)), (3,3))
        # print("First shape :- ",x.shape)
        x = F.max_pool2d(F.leaky_relu(self.conv2(x)), (2,2))
        # print("Second shape :- ",x.shape)
        x = F.max_pool2d(F.leaky_relu(self.conv3(x)), (3,3))
        # print("Third shape :- ",x.shape)

        if self.linear_shape is None:
            # print("X shape :- ",x[0].shape)
            self.linear_shape = x[0].shape[0] *  x[0].shape[1] *  x[0].shape[2]

        return x.to(get_device())
    
    def forward(self,x):
        x = self.convs(x)
        # print("Self Linear Shape :- ",self.linear_shape)
        x = x.view(-1,self.linear_shape)
        # print("Before feed to linear :- ",x.shape)
        x = F.normalize(x, p=2, dim=0)
        x = F.leaky_relu( self.linear1(x) )
        x = self.dropout(x)
        x = F.normalize(x, p=2, dim=0)
        # print("After 1 feed to linear :- ",x.shape)
        x = F.leaky_relu(self.linear2(x))
        x = self.dropout(x)
        x = F.normalize(x, p=2, dim=0)
        # print("Before Return :- ",x.shape)
        return self.output(x).to(get_device())
        
    

class FaceRecognition(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.embedding = CNN()
    def forward(self,x):

        return self.embedding(x).to(get_device())






# best_face_recognition_model.pth
# face_recognition_triplet2.pth
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)  # Batch Normalization after conv1
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128) # Batch Normalization after conv2
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256) # Batch Normalization after conv3
        self.dropout = nn.Dropout(p=0.20)

        self.linear_shape = None
        x = torch.randn(1, 3, 200, 200)  # Using batch size of 1 for shape calculation
        x = self.convs(x)
        self.linear1 = nn.Linear(self.linear_shape, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, 256)

    def convs(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))  # Apply Batch Norm and activation
        x = F.max_pool2d(x, (3, 3))
        x = F.leaky_relu(self.bn2(self.conv2(x)))  # Apply Batch Norm and activation
        x = F.max_pool2d(x, (2, 2))
        x = F.leaky_relu(self.bn3(self.conv3(x)))  # Apply Batch Norm and activation
        x = F.max_pool2d(x, (3, 3))

        if self.linear_shape is None:
            self.linear_shape = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

        return x.to(get_device())
    
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.linear_shape)
        x = F.normalize(x, p=2, dim=1)  # Normalize along the feature dimension
        x = F.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = F.normalize(x, p=2, dim=1)  # Normalize along the feature dimension
        x = F.leaky_relu(self.linear2(x))
        x = self.dropout(x)
        x = F.normalize(x, p=2, dim=1)  # Normalize along the feature dimension
        return self.output(x).to(get_device())
    
class FaceRecognition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = CNN()

    def forward(self, x):
        return self.embedding(x).to(get_device())
