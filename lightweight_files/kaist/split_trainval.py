import random

train_ratio = 0.8 # 80% training data for training, 20% for validation
with open('trainval.txt',mode='r') as trainval:
    samples = trainval.readlines()
    trainval.close()
train_num = int(train_ratio*len(samples))
train_samples = random.sample(samples,train_num)
val_samples = [i for i in samples if i not in train_samples]

with open('train.txt',mode='r') as train:
    train.writelines(train_samples)
    train.close()
    
with open('val.txt',mode='r') as val:
    val.writelines(val_samples)
    val.close()
