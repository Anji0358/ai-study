from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from  plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt

wine=load_wine()

pca=PCA(n_components=2)
lr=LogisticRegression(random_state=1,solver='lbfgs')


X=wine.data
y=wine.target

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.3,random_state=1,stratify=y
)

sc=StandardScaler()
sc.fit(X_train)

X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

X_train_pca=pca.fit_transform(X_train_std)
X_test_pca=pca.transform(X_test_std)
lr.fit(X_train_pca,y_train)
# plot_decision_regions(X_train_pca,y_train,classifier=lr)
plot_decision_regions(X_test_pca,y_test,classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()