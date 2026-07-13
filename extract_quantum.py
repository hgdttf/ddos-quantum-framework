#!/usr/bin/env python3
import re

with open('quantum_results_fresh.txt', 'r') as f:
    content = f.read()

pattern = r'QUANTUM SVM (\d+)-FOLD CROSS-VALIDATION \(([^)]+)\).*?AVERAGE ACROSS (\d+) VALID FOLDS:\s+Accuracy:\s+([\d.]+)\s+Precision:\s+([\d.]+)\s+Recall:\s+([\d.]+)\s+F1-Score:\s+([\d.]+)\s+Sensitivity:\s+([\d.]+)\s+Specificity:\s+([\d.]+)'

matches = re.findall(pattern, content, re.DOTALL)

print('QUANTUM SVM RESULTS')
print('=' * 90)
print('Dataset        k   Acc     Prec    Rec     F1      Sens    Spec   ')
print('-' * 90)

results = {}
for k, dataset, folds, acc, prec, rec, f1, sens, spec in matches:
    key = dataset.lower().replace(' ', '-') + '_k' + k
    results[key] = {
        'dataset': dataset.lower().replace(' ', '-'),
        'k': int(k),
        'accuracy': float(acc),
        'f1': float(f1)
    }
    print(dataset.ljust(15) + k.ljust(4) + acc.ljust(8) + prec.ljust(8) + rec.ljust(8) + f1.ljust(8) + sens.ljust(8) + spec.ljust(8))

print('-' * 90)
print('Total experiments: ' + str(len(results)))

print()
print('AVERAGES PER DATASET (k=2,3,4,5):')
print('-' * 40)
print('Dataset        Avg Acc  Avg F1  ')
print('-' * 40)

for dataset in ['nsl-kdd', 'unsw-nb15', 'cicddos2019', 'cicids2017', 'kddcup1999']:
    accs = [v['accuracy'] for k, v in results.items() if v['dataset'] == dataset]
    f1s = [v['f1'] for k, v in results.items() if v['dataset'] == dataset]
    if accs:
        avg_acc = sum(accs) / len(accs)
        avg_f1 = sum(f1s) / len(f1s)
        print(dataset.ljust(15) + str(round(avg_acc, 4)).ljust(9) + str(round(avg_f1, 4)))
