import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler

DATA_PATH='../data/...'

def load_and_standardize_data(data):
    df=data.iloc[:,1:]
    print(df.astype)
    df = df.values.reshape(-1, df.shape[1]).astype('float32')
    X_train, X_test = train_test_split(df, test_size=0.25, random_state=42)
    scaler = preprocessing.StandardScaler()
    X_train= scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test,scaler

from torch.utils.data import Dataset, DataLoader
class DataBuilder(Dataset):
    def __init__(self, train_data,train):
        if train:
            self.x = torch.from_numpy(train_data)
            self.len=self.x.shape[0]
            print(self.x.shape)
        else:
            self.x = torch.from_numpy(train_data)
            self.len=self.x.shape[0]
            print(self.x.shape)
    def __getitem__(self,index):
        return self.x[index]
    def __len__(self):
        return self.len

one_train = train_data.iloc[np.where(train_data[0]==1)[0]]
zero_train =train_data.iloc[np.where(train_data[0]==0)[0]]

X_train1,X_test1,standardizer1 = load_and_standardize_data(one_train)
traindata_one=DataBuilder(X_train1,train=True)
testdata_one=DataBuilder(X_test1, train=False)
trainloader1=DataLoader(dataset=traindata_one,batch_size=32)
testloader1=DataLoader(dataset=testdata_one,batch_size=32)

X_train2,X_test2,standardizer2 = load_and_standardize_data(zero_train)
traindata_zero=DataBuilder(X_train2,train=True)
testdata_zero=DataBuilder(X_test2, train=False)
trainloader2=DataLoader(dataset=traindata_zero,batch_size=32)
testloader2=DataLoader(dataset=testdata_zero,batch_size=32)


class Autoencoder(nn.Module):
    def __init__(self, D_in, H, H2, latent_dim):

        # Encoder
        super(Autoencoder, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)
        self.linear2 = nn.Linear(H, H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3 = nn.Linear(H2, H2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H2)

        #         # Latent vectors mu and sigma
        self.fc1 = nn.Linear(H2, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

        #         # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H2)
        self.fc_bn4 = nn.BatchNorm1d(H2)

        #         # Decoder
        self.linear4 = nn.Linear(H2, H2)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H2)
        self.linear5 = nn.Linear(H2, H)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H)
        self.linear6 = nn.Linear(H, D_in)
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)

        self.relu = nn.ReLU()

    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)

        return r1, r2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        return self.lin_bn6(self.linear6(lin5))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # self.decode(z) ist später recon_batch, mu ist mu und logvar ist logvar
        return self.decode(z), mu, logvar


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    # x_recon ist der im forward im Model erstellte recon_batch, x ist der originale x Batch, mu ist mu und logvar ist logvar
    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD

# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

D_in = trainloader1.dataset.x.shape[1]
print(D_in)
H = 50
H2 =12
latent_dim=10
model = Autoencoder(D_in, H, H2,latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 4000
log_interval = 50
val_losses = []
train_losses1 = []
test_losses1 = []
train_losses2 = []
test_losses2 = []

def train(epoch,trainloader,train_losses,i):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(trainloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_mse(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    if epoch % 200 == 0:
        print('====> Epoch: {} Average training loss: {:.4f}'.format(
            epoch, train_loss / len(trainloader.dataset)))
        train_losses.append(train_loss / len(trainloader.dataset))
    if(i==1):
        train_losses1=train_losses
    else:
        train_losses2=train_losses

def test(epoch,testloader,test_losses,i):
    with torch.no_grad():
        test_loss = 0
        for batch_idx, data in enumerate(testloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_mse(recon_batch, data, mu, logvar)
            test_loss += loss.item()
        if epoch % 200 == 0:
            print('====> Epoch: {} Average test loss: {:.4f}'.format(
                    epoch, test_loss / len(testloader.dataset)))
            test_losses.append(test_loss / len(testloader.dataset))
    if(i==1):
        test_losses1=test_losses
    else:
        test_losses2=test_losses


for epoch in range(1, epochs + 1):
    train(epoch,trainloader1,train_losses1,1)
    test(epoch,testloader1,test_losses1,1)

for epoch in range(1, epochs + 1):
    train(epoch,trainloader2,train_losses2,0)
    test(epoch,testloader2,test_losses2,0)

with torch.no_grad():
    for batch_idx, data in enumerate(testloader1):
        data1 = data.to(device)
        optimizer.zero_grad()
        recon_batch1, mu1, logvar1 = model(data1)
    for batch_idx, data in enumerate(testloader2):
        data2 = data.to(device)
        optimizer.zero_grad()
        recon_batch2, mu2, logvar2 = model(data2)

sigma1 = torch.exp(logvar1/2)
sigma2 = torch.exp(logvar2/2)
no_samples = 115
q1 = torch.distributions.Normal(mu1.mean(axis=0), sigma1.mean(axis=0))
z1 = q1.rsample(sample_shape=torch.Size([no_samples]))
q2 = torch.distributions.Normal(mu2.mean(axis=0), sigma2.mean(axis=0))
z2 = q2.rsample(sample_shape=torch.Size([no_samples]))
with torch.no_grad():
    pred1 = model.decode(z1).cpu().numpy()
    pred2 = model.decode(z2).cpu().numpy()

fake_data1 = standardizer1.inverse_transform(pred1)
fake_data2 = standardizer2.inverse_transform(pred2)