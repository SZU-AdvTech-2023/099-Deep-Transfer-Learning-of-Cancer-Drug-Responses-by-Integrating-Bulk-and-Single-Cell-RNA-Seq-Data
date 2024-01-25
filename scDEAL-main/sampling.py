from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


#upsampling
def upsampling(X_train,Y_train):
    ros = RandomOverSampler(random_state=42)  # 通过随机复制少数类别的样本，增加其数量，从而平衡各个类别的样本数量
    X_train, Y_train = ros.fit_resample(X_train, Y_train)  # 通过过采样的方式增加少数类别的样本，使得各个类别的样本数量更加平衡
    return X_train,Y_train

#downsampling
def downsampling(X_train,Y_train):
    rds = RandomUnderSampler(random_state=42)  # 通过随机去除多数类别的样本，以使得各个类别的样本数量更加平衡
    X_train, Y_train = rds.fit_resample(X_train, Y_train)
    return X_train,Y_train

## nosampling
def nosampling(X_train,Y_train):
    return X_train,Y_train

##SOMTE
def SMOTEsampling(X_train,Y_train):
    sm=SMOTE(random_state=42)  # 在特征空间中合成（生成）新的少数类别样本
    X_train, Y_train = sm.fit_resample(X_train, Y_train)
    return X_train,Y_train
