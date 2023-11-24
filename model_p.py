from torchinfo import summary
from models.RemoteNet import RemoteNet
from models.Encoder import Encoder

model = RemoteNet(dim=64,dims=(32,64,160,256),num_classes=6)

summary(model,input_size=(1,3,256,256)) 