import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_score,recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.sampler import RandomSampler,SequentialSampler
from torch.optim.lr_scheduler import MultiStepLR

#outlier removal function
def outlier_removal(data_temp,label_temp):
    idx=data_temp<3
    #print(np.sum(np.sum(idx,axis=1)>278))
    rows_idxs=np.sum(idx,axis=1)>278        #Parameter to determine the outliers
    data_temp=data_temp[rows_idxs,:]
    label_temp=label_temp[rows_idxs]

    return data_temp,label_temp

def class_distribution(labels):
    unique_classes = set(labels)
    for i in unique_classes:
        idx = labels == i
        print('Class ', i, ' examples count:', np.sum(idx))


#Removing columns contains all entries as mean
def zeros_std_column_removal(temp_train,temp_test):
    stds=np.std(temp_train,axis=0,dtype=np.float32)
    #print("0 std columns: ",np.where(stds==0))
    temp_train=temp_train[:,(stds!=0)]
    temp_test= temp_test[:, (stds != 0)]
    print('Zero standard deviation column removed!')
    return temp_train,temp_test


#Removing columns contains all entries as 0
def zero_column_removal(temp_train,temp_test):
    idx=(temp_train!=0).any(axis=0)
    #print(temp.shape)
    temp_train=temp_train[:,idx]
    temp_test = temp_test[:, idx]
    print('Zero columns removed!')
    return temp_train,temp_test


#standard deviation calculation
def std_values(temp):
    stds=np.std(temp,axis=0,dtype=np.float32)
    print(np.sort(stds))

#Applying LDA to reduce dimentionalilty
def apply_lda(train_X,train_Y,test_X):
    train_Y_en=label_encoder(np.copy(train_Y))
    lda=LinearDiscriminantAnalysis(n_components=160)    #Number of LDA-Dimensions
    lda.fit(train_X,train_Y_en)
    train_X=lda.transform(train_X)
    test_X=lda.transform(test_X)
    return train_X,test_X






#Applying PCA to reduce dimensionality
def apply_pca(temp_train,temp_test):
    pca=PCA(n_components=160)       #Number of PCA-Dimensions
    pca.fit(temp_train)
    temp_train=pca.transform(temp_train)
    temp_test=pca.transform(temp_test)
    return temp_train,temp_test

#Normalizing the data to 0 mean and 1 std
def normalization(temp_train,temp_test):
    means=np.mean(temp_train,axis=0)
    stds=np.std(temp_train,axis=0,dtype=np.float32)
    #print(np.where(stds==0))
    temp_train=temp_train-means
    temp_train=temp_train/stds
    temp_test=temp_test-means
    temp_test=temp_test/stds
    print('Normalization done!')

    return temp_train,temp_test



#Dividing the data in training and testing sets
def splitting_data(data,labels):
    train_X,test_X,train_Y,test_Y=train_test_split(data,labels,test_size=0.3,random_state=10)
    print('Data split in training and testing sets')
    return train_X,train_Y,test_X,test_Y


#reading the file
def read_file(file_name):
    df=pandas.read_csv(file_name,delimiter=',',header=None)
    df=df.as_matrix()
    data=df[:,:-1]
    labels=df[:,-1]
    print('File Read!')
    return data.astype(np.float32),labels


def logistic_regression_classifier(X,Y):
    clf=LogisticRegression(random_state=0, multi_class='multinomial',solver='lbfgs',max_iter=200)
    clf.fit(X,Y)
    return clf




def predict(clf,test_X,test_Y):
    preds=clf.predict(test_X)
    precision=precision_score(test_Y,preds,average='weighted')
    recall=recall_score(test_Y,preds,average='weighted')
    print('Precision:',precision)
    print('Recall:', recall)


def label_encoder(temp):
    labels = {'A', 'B', 'C', 'D', 'E'}
    temp_label=np.copy(temp)
    j=0
    for i in labels:
        idx=temp_label==i
        temp_label[idx]=j
        j=j+1
    return temp_label.astype(np.float64)

def svm_classifier(X,Y):
    clf=svm.LinearSVC()
    #clf=svm.SVC(kernel='poly')
    clf.fit(X,Y)
    return clf



def random_forest_classifier(X,Y):
    clf=RandomForestClassifier(n_estimators=100,max_depth=10)      #Hyperparameter for Random Forests
    clf.fit(X,Y)
    return clf


class nn_classifier(nn.Module):
    def __init__(self,input_dim,num_classes):
        super(nn_classifier,self).__init__()
        self.s1=nn.Sequential(
            nn.Linear(input_dim,350),
            nn.ReLU(),
            nn.Linear(350,450),
            nn.ReLU(),
            nn.Linear(450,300),
            nn.ReLU(),
            nn.Linear(300,num_classes)
        )

    def forward(self,X):
        res=self.s1(X)
        return res



def nn_classifier_setup(X,Y):
    criterion = nn.CrossEntropyLoss()
    # criterion=nn.L1Loss()
    input_dim=X.shape[1]
    num_classes=len(set(Y))
    net = nn_classifier(input_dim,num_classes)

    X=torch.Tensor(X)
    Y=torch.LongTensor(Y)
    ds = torch.utils.data.TensorDataset(X,Y)
    Rsampler=RandomSampler(ds)
    dl = DataLoader(ds, batch_size=1000,sampler=Rsampler)

    learning_rate = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=[70, 120, 600, 800], gamma=0.01) #Adaptive learning rate

    i = 0
    num_epochs = 1000
    loss_min=-1


    for j in range(num_epochs):
        print('Epoch:', j)
        for data,label in dl:
            optimizer.zero_grad()
            data=Variable(data)
            label=Variable(label)
            output = net(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()


            print('Loss:', loss.data[0])
        print()
        print('############################')

    return net






def nn_predict(clf,X,Y):
    X = torch.Tensor(X)
    Y_tar=torch.LongTensor(Y)
    ds = torch.utils.data.TensorDataset(X,Y_tar)
    Ssampler = SequentialSampler(ds)
    bs=1000
    dl = DataLoader(ds, batch_size=bs, sampler=Ssampler)
    precision=0.0
    recall=0.0
    i=0
    acc=0
    for data,label in dl:
        i=i+1
        data=Variable(data)
        output = clf(data)
        precision=precision+(data.data.shape[0]/bs)*(precision_score(label.numpy(),torch.max(output,1)[1].data.numpy(),average='weighted'))
        recall= recall+(data.data.shape[0]/bs)*(recall_score(label.numpy(), torch.max(output, 1)[1].data.numpy(), average='weighted'))

    print('Precision:',precision/i)
    print('Recall:',recall/i)



class autoencoder(nn.Module):
    def __init__(self,input_dim):
        super(autoencoder, self).__init__()
        self.input_dim=input_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 250),
            nn.ReLU(),
            nn.Linear(250, 200),
            nn.ReLU(), nn.Linear(200, 50))
        self.decoder = nn.Sequential(
            nn.Linear(50, 200),
            nn.ReLU(),
            nn.Linear(200, 250),
            nn.ReLU(), nn.Linear(250, self.input_dim))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self,X):
        en=self.encoder(X)
        return en

class custom_dataset(Dataset):
    def __init__(self,data):
        self.data=data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.Tensor(self.data[idx])

def autoencoder_setup(X):

    criterion = nn.MSELoss()
    # criterion=nn.L1Loss()
    input_dim=X.shape[1]
    net = autoencoder(input_dim)

    ds = custom_dataset(X)
    Rsampler=RandomSampler(ds)
    dl = DataLoader(ds, batch_size=200,sampler=Rsampler)

    learning_rate = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    #scheduler = MultiStepLR(optimizer, milestones=[70, 120], gamma=0.01)

    i = 0
    num_epochs = 1000
    loss_min=-1


    for j in range(num_epochs):
        #print('Epoch:', j)
        for data in dl:
            optimizer.zero_grad()
            data=Variable(data)
            output = net(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()


            #print('Loss:', loss.data[0])
        #print()
        #print('############################')

    return net


def autoencoder_dim_reduction(train_X,test_X):
    net=autoencoder_setup(train_X)
    train_X=net.encode(Variable(torch.Tensor(train_X)))
    test_X=net.encode(Variable(torch.Tensor(test_X)))

    return train_X,test_X


def zero_rows(X):
    idx=X==0
    print('Zero Rows:',np.sum(idx.all(axis=1)))


def autoencoder_model_validation(X):
    criterion = nn.MSELoss()
    # criterion=nn.L1Loss()
    input_dim = X.shape[1]
    net = autoencoder(input_dim)
    data=Variable(torch.Tensor(X[100:150]))

    learning_rate = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler=MultiStepLR(optimizer,milestones=[100,300,400],gamma=0.01)

    i = 0
    num_epochs = 500
    loss_min=100
    epoch_num=0
    for j in range(num_epochs):
        #print('Epoch:', j)

        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

        if(loss.data[0]<loss_min):
            loss_min=loss.data[0]
            epoch_num=j
        #print('Loss:', loss.data[0])
        #print()
        #print('############################')
    print('Minimum Loss:',loss_min)



def preliminary_steps():
    file_name='E:/sample.csv'  #Filename
    data, labels = read_file(file_name)     #File reading function
    train_X, train_Y, test_X, test_Y = splitting_data(data, labels)     #data spliting function
    zero_rows(train_X)       #Printing if there are zero-rows in data

    train_X, test_X = zero_column_removal(train_X, test_X)      #removing the columns with all 0 entries

    train_X, test_X = zeros_std_column_removal(train_X, test_X)     #Removing the columns with 0 standard deviation
    train_X, test_X = normalization(train_X, test_X)                #Normalizing data to 0 mean and 1 std

    train_X, train_Y = outlier_removal(train_X, train_Y)            #Removing outliers from training set
    test_X, test_Y = outlier_removal(test_X, test_Y)                #Removing outliers from test set
    print('Outliers removed!')
    return train_X,train_Y,test_X,test_Y






def main():

    train_X,train_Y,test_X,test_Y=preliminary_steps()
    train_X_pca,test_X_pca=apply_pca(train_X,test_X)
    print('PCA applied!')
    train_X_lda,test_X_lda=apply_lda(train_X,train_Y,test_X)
    print('LDA applied!')
    train_X_auto, test_X_auto = autoencoder_dim_reduction(train_X, test_X)
    print('Auto-encoder applied!')

    train_Y_en = label_encoder(train_Y)  # Encoding labels for SVM classifier
    test_Y_en = label_encoder(test_Y)  # Encoding labels for SVM classifier
    print('Label encoding applied!')

    print('Auto-encoder Validation!')
    autoencoder_model_validation(train_X)



    #appling logistic regression with PCA, LDA and auto encoded data

    print('Logistic Regression with PCA:')
    clf=logistic_regression_classifier(train_X_pca,train_Y)
    predict(clf,test_X_pca,test_Y)
    
    print('Logistic Regression with LDA:')
    clf = logistic_regression_classifier(train_X_lda, train_Y)
    predict(clf, test_X_lda, test_Y)

    #print('Logistic Regression with Auto-encoder:')
    #clf=logistic_regression_classifier(train_X_auto,train_Y)
    #predict(clf,test_X_auto,test_Y)


    #Applying SVM
    
    print('SVM with PCA:')
    clf=svm_classifier(train_X_pca,train_Y_en)
    predict(clf,test_X_pca,test_Y_en)

    print('SVM with LDA:')
    clf = svm_classifier(train_X_lda, train_Y_en)
    predict(clf, test_X_lda, test_Y_en)

    #print('SVM with Auto-encoder:')
    #clf = svm_classifier(train_X_auto, train_Y_en)
    #predict(clf, test_X_auto, test_Y_en)


    
    # appling Random Forest with PCA, LDA and auto encoded data
    
    print('Random Forest with PCA:')
    clf = random_forest_classifier(train_X_pca, train_Y_en)
    predict(clf, test_X_pca, test_Y_en)

    print('Random Forest with LDA:')
    clf = random_forest_classifier(train_X_lda, train_Y_en)
    predict(clf, test_X_lda, test_Y_en)

    #print('Random Forest with Auto-encoder:')
    #clf = random_forest_classifier(train_X_auto, train_Y)
    #predict(clf, test_X_auto, test_Y)
    

    #Applying Neural Network wih PCA,LDA and auto encoded data

    print('Neural Network with PCA:')
    clf = nn_classifier_setup(train_X_pca, train_Y_en)
    nn_predict(clf, test_X_pca, test_Y_en)

    print('Neural Network with LDA:')
    clf=nn_classifier_setup(train_X_lda,train_Y_en)
    nn_predict(clf,test_X_lda,test_Y_en)

    #print('Neural Network with Auto-encoder:')
    #clf = nn_classifier_setup(train_X_auto, train_Y_en)
    #nn_predict(clf, test_X_auto, test_Y_en)


if __name__=="__main__":
    main()