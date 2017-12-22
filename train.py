#import os
#import numpy as np
#import pickle
#from PIL import Image
#import feature
#x=[]
#path_face = "C:/Users/asus/project3/datasets/original/face/"
#for img_face in os.listdir(path_face):
#    x.append(feature.NPDFeature.extract(feature.NPDFeature(np.array(Image.open(path_face+img_face).resize((24, 24)).convert('L')))))
#path_nonface = "C:/Users/asus/project3/datasets/original/nonface/"
#for img_nonface in os.listdir(path_nonface):
#    x.append(feature.NPDFeature.extract(feature.NPDFeature(np.array(Image.open(path_nonface+img_nonface).resize((24, 24)).convert('L')))))
#pickle.dump(np.array(x),open('datasets.pkl','wb'))
import numpy as np
import pickle
import ensemble
from sklearn.cross_validation import train_test_split 
from sklearn.tree import DecisionTreeClassifier
x=pickle.load(open('datasets.pkl','rb'))
y=np.ones(1000)
y[500:999]=-1
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)
DTclf = DecisionTreeClassifier(max_depth=4)
ADBTclf=ensemble.AdaBoostClassifier(DTclf,20).fit(x_train,y_train)
y_predict=ensemble.AdaBoostClassifier.predict(ADBTclf,x_test,0)
print(1-np.sum(y_predict!=y_test)/np.shape(x_test)[0])
