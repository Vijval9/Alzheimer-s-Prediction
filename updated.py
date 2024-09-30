# Made some changes to the training part. Fixed data leakage and got higher accuracy.
# Please note that this is just the code for the training part, after preprocessing was done and features were extracted.

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
l = []

for i in df3['cat']:
    if(i!=0):
        l.append(1)
    else:
        l.append(i)

df3['cat'] = l
df3.drop(axis=1,columns = gen,inplace=True)        # finally have 121 degs
fina = df3

X = fina[df3.columns[13:]]
X.head()
print(X.shape)
y = df3['cat']
y.head()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=41)

gnb = GaussianNB()
knn = KNeighborsClassifier()
logreg = LogisticRegression(solver='saga',random_state=41)
rf = RandomForestClassifier(random_state=41,n_estimators=100,max_features=400)
abc = AdaBoostClassifier(random_state=41, n_estimators=250)
gbc = GradientBoostingClassifier(random_state=41,max_features=60)
svc = SVC(random_state=41,probability=True)

models = [gnb,knn,logreg,rf,abc,svc]

d={}

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=41)

ss = StandardScaler()
pca = PCA(n_components=50)

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

results = []
    
    
for i in models:
    print(i,": ",end=" ")
    i.fit(X_train,y_train)
    preds = i.predict(X_test)
    
    acc = accuracy_score(y_test,preds)
    prec = precision_score(y_test,preds)
    rec = recall_score(y_test,preds)
    f1 = f1_score(y_test,preds)
    
    results.append([i.__class__.__name__,acc,prec,rec,f1])
        
    
    print("Accuracy: ",acc)
    
results = pd.DataFrame(columns = ['Model','Accuracy','Precision','Recall','F1 Score'],data=results)

    
