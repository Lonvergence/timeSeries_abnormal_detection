import matplotlib.pyplot as plt
import time
import numpy as np

def draw_loss(train_loss, val_loss, stride=10, path = './'):
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    
    if(len(train_loss) != len(val_loss)):
        return
    
    epoch = range(0, len(train_loss), stride)
    
    tr_loss = train_loss[epoch]
    va_loss = val_loss[epoch]
    
    plt.plot(epoch,tr_loss,'k-',label='train_loss')
    plt.plot(epoch,va_loss,'r:',label='val_loss')
    
    plt.savefig(f"{path}{int(time.time())}.png")
    
    