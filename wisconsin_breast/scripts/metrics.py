import numpy as np
from sys import argv
import csv

if len(argv) != 3:
    print "USAGE: python metrics.py in_file out_file"
    exit(0)

_, data_file, out_file = argv

with open(data_file, 'rb') as f:
    reader = csv.reader(f)
    header = next(reader)
    data = []
    for row in reader:
        data.append(row)
    data = np.array(data)

labels = list(data[:,1])
labels = [0 if x == "M" else 1 for x in labels]

data = np.insert(data[:,2:], 0, data[:,0], axis=1)
data = data.astype(np.float)
header = [header[0]] + header[2:]

means = []
variances = []
correlations = []
for c in range(0,len(data[0])):
    means.append(np.mean(data[:,c]))
    variances.append(np.var(data[:,c]))
    correlations.append(np.corrcoef(data[:,c],labels))

with open(out_file, 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(["metric"]+header)
    writer.writerow(["mean"] + means)
    writer.writerow(["variance"] + variances)
    writer.writerow(["correlation to labels"] + [c[0][1] for c in correlations])
