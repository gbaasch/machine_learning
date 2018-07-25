from ml_utils.parse_data import parse_data


fi = "/Users/Brendan/Documents/GitHub/machine_learning/wisconsin_breast/train.csv"
print "huh"
with open(fi, "r") as f:
    print "this"
    data, labels, header = parse_data(f, 1, "M")

    print sum(labels)
    print len(labels)

print "ok"