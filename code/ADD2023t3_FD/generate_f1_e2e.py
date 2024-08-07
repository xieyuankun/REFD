import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np


ood_result = torch.load('./ood_step/result_oodscore.pt')
print(str(result[0].item()))

count = 0
with open('result/result_FD.txt', 'r') as f2:
    with open('result/result_ood.txt', 'w') as f3:
        for line in f2:
            file = line.strip().split()[0]
            category = line.strip().split()[1]
            score = line.strip().split()[2]
            ood = ood_result[count].item()
            if float(ood) < 0:    # Adjusting the threshold to determine OOD !!
                category = 7

            f3.write(f'{file} {category} {score} {ood}\n')
            count+=1

# 2
realdict = {}
ooddict1 = {}

with open('result/result_RE.txt', 'r') as f1:
    for line in f1:
        file = line.strip().split()[0]
        category = line.strip().split()[1]
        score = line.strip().split()[2]
        if int(category) == 1 and float(score) > 0.98:   # Tight OC-Softmax threshold
            realdict[file] = category


'''
fusion RE and FD
'''

with open('result/result_ood.txt', 'r') as f2:
    with open('result/result_REFD.txt', 'w') as f3:
        for line in f2:
            file = line.strip().split()[0]
            category = line.strip().split()[1]
            score = line.strip().split()[2]
            if file in realdict.keys():
                f3.write(f'{file} {0}\n')
            else:
                f3.write(f'{file} {category}\n')

# 3
output_file = 'result/result_REFD.txt'
label_file = 'eval.txt'             # ADD2023 Track3 label

with open(output_file, 'r') as f:
    output_lines = f.read().splitlines()

with open(label_file, 'r') as f:
    label_lines = f.read().splitlines()

# Extract true labels and predicted labels
true_labels = [int(line.split()[1]) for line in label_lines]
predicted_labels = [int(line.split()[1]) for line in output_lines]

# Compute the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Calculate the overall F1 score
f1_total = f1_score(true_labels, predicted_labels, average='macro')

# Calculate F1 scores for each class
f1_per_class = f1_score(true_labels, predicted_labels, average=None)

# Print results
print("Overall F1 score:", f1_total)
for class_id, f1 in enumerate(f1_per_class):
    print(f"F1 score - Class {class_id}: {f1}")