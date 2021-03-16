import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

names = ["NB-SKLEAEN - E.W", "NB-SKLEARN  - E.F", "NB-SELF - E.W", "NB-SELF - E.F", "ID3-SKLEARN - EW",
         "ID3-SKLEARN - E.F", "ID3-SELF - E.W", "ID3-SELF - E.F", "KNN - E.W", "KNN - E.F"]
fpr = [0.48, 0.78, 0.45, 0.45, 0.41, 0.39, 0.41, 0.41, 0.35, 0.29]
tpr = [0.58, 0.48, 0.66, 0.67, 0.51, 0.52, 0.51, 0.51, 0.51, 0.51]
c=['black', 'blue', 'green', 'red', 'purple', 'pink', 'yellow', 'orange', "brown", "gray"]
# line = plt.scatter(fpr, tpr, marker="o", c=['black', 'blue', 'green', 'red', 'purple', 'pink', 'yellow', 'orange',
# "brown", "gray"],cmap=names)

for f,t,c,label in zip(fpr,tpr,c,names):
    plt.scatter(f,t,c=c, label=label)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()
