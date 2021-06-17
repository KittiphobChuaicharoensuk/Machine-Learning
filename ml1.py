#การแสดงภาพด้วย Pylab และ Matplotlib 

import matplotlib.pyplot as plt
from sklearn import datasets
digit_dataset=datasets.load_digits() #ไลบารี่MNIST
print(digit_dataset.target[2])
plt.imshow(digit_dataset.images[2],cmap=plt.get_cmap("gray"))
plt.show()