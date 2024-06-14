# from sklearn.metrics import f1_score
# from collections import defaultdict
#
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np

output_file = './result/result_REFD.txt'
label_file = './label/eval.txt'

with open(output_file, 'r') as f:
    output_lines = f.read().splitlines()

with open(label_file, 'r') as f:
    label_lines = f.read().splitlines()

true_labels = [int(line.split()[1]) for line in label_lines]
predicted_labels = [int(line.split()[1]) for line in output_lines]
conf_matrix = confusion_matrix(true_labels, predicted_labels)
f1_total = f1_score(true_labels, predicted_labels, average='macro')
f1_per_class = f1_score(true_labels, predicted_labels, average=None)

print("总体F1分数:", f1_total)
for class_id, f1 in enumerate(f1_per_class):
    print(f"F1分数 - 类别 {class_id}: {f1}")

