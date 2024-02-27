import numpy as np
import csv

import matplotlib.pyplot as plt
import math


comment = "l2_drp_"

train_ls = []
val_ls = []

step1 = []
step2 = []

train_file_name = "./run-" + str(comment) + "_Ls_trn-tag-.csv"
val_file_name = "./run-" + str(comment) + "_Ls_val-tag-.csv"

with open(train_file_name) as csv_file:

	csv_reader = csv.reader(csv_file)

	for i, row in enumerate(csv_reader):

		if i >0:
			#print(row)

			step1.append(int(row[1]))
			train_ls.append(float(row[2]) ** 0.5)
			#train_ls.append(math.log(float(row[2]) ** 0.5, 10))



#print(step, train_ls)


with open(val_file_name) as csv_file:

	csv_reader = csv.reader(csv_file)

	for i, row in enumerate(csv_reader):

		if i >0:
			#print(row)

			step2.append(int(row[1]))
			val_ls.append(float(row[2]) ** 0.5)
			#val_ls.append(math.log(float(row[2]) ** 0.5, 10))


assert step1 == step2

#print(step1)

#print()

#print(step2)

plt.plot(step1, train_ls, color = 'r')
plt.plot(step2, val_ls, color = 'g')
plt.yscale('log')
#plt.yticks([10 ** -1, 10 ** -.5, 10 ** -.25])

plt.legend(["Train", "Validation"])
plt.xlabel('Epochs')
plt.ylabel('RMSE Loss (log scale)')



plt.savefig(comment + '.png', dpi = 350)



plt.show()

