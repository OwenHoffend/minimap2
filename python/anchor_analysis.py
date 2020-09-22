import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt

class Query:
    def __init__(self):
        self.x, self.y = [], []
        self.name = ""

    def merge(self, query_list):
        for query in query_list:
            self.x += query.x
            self.y += query.y

def load_anchors(fn, num_quers):
    try:
        with open(fn) as f:
            q_cnt = 0
            queries = []
            while True:
                line = f.readline()
                if line.startswith("@") or line == "":
                    if q_cnt != 0:
                        queries.append(q)
                    else:
                        ref_name = line.split(',')[0][1:]
                    if q_cnt == num_quers:
                        break
                    if line == "":
                        print("Warning: Only read {} out of {} queries" \
                        .format(q_cnt, num_quers))
                        break
                    q = Query()  
                    q_cnt = q_cnt + 1
                    q.name = line.split(',')[1].strip()
                else:
                    ls = line.split(',')
                    q.x.append(int(ls[0], 16))
                    q.y.append(int(ls[1], 16))
    except IOError as e:
        print(e)
        exit()
    return queries, ref_name

def plot_chain_lengths(qd):
    x, y = np.array(qd.x), np.array(qd.y)
    diffs = np.subtract(x, y)
    uniques = np.unique(diffs, return_counts=True)
    plt.scatter(uniques[0], np.log10(uniques[1]))
    plt.xlabel("delta_c from y=x line")
    plt.ylabel("Number of points (log scale)")
    plt.title("Chain length distribution")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot anchor output from minimap2. \
            Created by Owen Hoffend.")
    parser.add_argument("fn", help="File with anchor coordinates exported from minimap2")
    parser.add_argument("-n", type=int, default=1, help="Number of queries to read from the file")
    parser.add_argument("--qrange", nargs='+', type=int, help="Query index range, as a python list")
    args = parser.parse_args()
    fn, num_quers = args.fn, args.n
    queries, ref_name = load_anchors(fn, num_quers)

    merged_queries = Query()
    merged_queries.merge(queries)
    plot_chain_lengths(merged_queries)

    fig, ax = plt.subplots()
    qrange = args.qrange
    if qrange != None:
        for i in qrange:
            q = queries[i-1]
            plt.scatter(q.x, q.y, s=10, label=q.name)
    else:
        for q in queries:
            plt.scatter(q.x, q.y, s=10, label=q.name)

    ax.legend(loc='lower left', bbox_to_anchor= (0.0, 1.06), ncol=2,
                borderaxespad=0, frameon=False)
    plt.xlabel("Reference Position")
    plt.ylabel("Query Position")
    plt.title("Ref: " + ref_name)
    plt.show()