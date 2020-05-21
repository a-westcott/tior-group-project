import csv, sys
from collections import defaultdict as dd

def make_table(filepath):
    with open(filepath) as f:
        count = dd(int)
        iterations = dd(int)
        time = dd(float)
        keys = set()
        reader = csv.reader(f)
        for row in reader:
            f, x0, x1, x2, x3, i, t = row
            key = (f, x0, x1, x2, x3)
            count[key] += 1
            iterations[key] += int(i)
            time[key] += float(t)
            keys.add(key)
    processed_data = []
    for key in keys:
        processed_data.append((*key, count[key], iterations[key]/count[key], time[key]/count[key]))
    processed_data.sort(key=lambda x: (float(x[0]), -int(x[5])))

    with open(sys.argv[1][3:-4]+ '-table.txt', 'w') as table:
        table.write("\\begin{center}\\begin{tabular}{ccccc}\n")
        table.write("\t$f$ value & Minimiser & No. of times & Av. iterations & Av. search time (sec) \\\\ \\hline\n")
        for f, x0, x1, x2, x3, n, i, t in processed_data:
            table.write(f"\t{f} & ({x0}, {x1}, {x2}, {x3}) & {n} & {float(i):.2f} & {float(t):.6f} \\\\\n")
        table.write("\\end{tabular}\\end{center}\n")

def main():
    make_table(sys.argv[1])


if __name__ == '__main__':
    main()