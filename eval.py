import torch
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

def label_classification(z, y, ratio):
    z = z.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    
    norm_z = normalize(z)
    x_train, x_test, y_train, y_test = train_test_split(norm_z,y,test_size=1-ratio)
       
    # for small dataset, liblinear solver is a good choice
    lreg = LogisticRegression(multi_class='multinomial')
    gbc = GradientBoostingClassifier()
    
    lreg.fit(x_train, y_train)
    gbc.fit(x_train, y_train)
    
    lreg_pred = lreg.predict(x_test)
    gbc_pred = gbc.predict(x_test)
    
    lreg_acc = (lreg_pred == y_test).sum() / len(y_test)
    gbc_acc = (gbc_pred == y_test).sum() / len(y_test)
    
    return {
        'lreg': lreg_acc,
        'gbc': gbc_acc
    }

