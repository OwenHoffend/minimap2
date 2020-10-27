import sys, os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from bloom_filter import BloomFilter

BHAM_DIR = ".\\minimap2\\test\\Bham_decompressed"

class Query:
    def __init__(self):
        self.x, self.y = [], []
        self.name = ""
        self.score = 0

    def merge(self, query_list):
        for query in query_list:
            self.x += query.x
            self.y += query.y

def read_quality(filedir):
    q_scores = {}
    for fn in os.listdir(filedir):
        fp = os.path.join(filedir, fn)
        try:
            with open(fp) as f:
                is_qs = False
                name = ''
                for line in f.readlines():
                    if is_qs:
                        q_scores[name] = sum([ord(c) for c in line.strip()]) / len(line.strip())
                        is_qs = False
                    elif line.startswith('+'):
                        is_qs = True #Next line is a quality score
                    elif line.startswith('@'):
                        name = line.split(' ')[0][1:]
                        is_qs = False

        except IOError as e:
            print(e)
            exit()
    return q_scores

def load_anchors(fn, num_quers, readall, q_scores=None):
    try:
        with open(fn) as f:
            q_cnt = 0
            queries = []
            header_last = False
            while True:
                line = f.readline()
                if line.startswith("@") or line == "":
                    if q_cnt != 0 and not header_last:
                        queries.append(q)
                    if line == "" or (not readall and q_cnt == num_quers):
                        break
                    q = Query()  
                    q_cnt = q_cnt + 1
                    q.name = line[1:].strip()
                    if q_scores != None:
                        q.score = q_scores[q.name]
                    header_last = True
                else:
                    ls = line.split(',')
                    q.x.append(int(ls[0], 16))
                    q.y.append(int(ls[1], 16))
                    header_last = False
    except IOError as e:
        print(e)
        exit()
    return queries

def plot_chain_lengths(qd):
    x, y = np.array(qd.x), np.array(qd.y)
    diffs = np.subtract(x, y)
    uniques = np.unique(diffs, return_counts=True)

    #Find the primary diagonal
    max_ind = np.argmax(uniques[1])
    max_val = uniques[0][max_ind]

    plt.scatter(np.subtract(uniques[0], max_val), np.log10(uniques[1]))
    plt.xlabel("delta_c from primary diagonal")
    plt.ylabel("Number of points (log scale)")
    plt.title("Chain length distribution")
    plt.show()

def compression_ratio_analysis(fn, queries, plotrange=300):
    avgs = np.zeros(plotrange)
    avg_gt_5 = 0
    avg_comp = 0
    avg_comp_no_sing = 0
    for sr in range(10):
        for query in queries:
            x, y = np.array(query.x), np.array(query.y)
            #diffs = np.subtract(x, y)
            diffs = np.left_shift(np.right_shift(np.subtract(x, y), sr), sr)
            unique_cnts = np.unique(diffs, return_counts=True)[1]
            sorted_cnts = np.sort(unique_cnts)[-plotrange:]
            scaled_cnts = np.true_divide(sorted_cnts, np.size(x))
            #Some other stats
            #Average number of chains 5 or longer
            num_chains = len(list(filter(lambda x: x >= 5, unique_cnts)))
            avg_gt_5 += num_chains

            #Compression ratio = number of unique_cnts / number of points
            compression_ratio = len(unique_cnts) / np.size(x)
            avg_comp += compression_ratio

            #Compression ratio with singletons removed
            compr_no_sing = len(list(filter(lambda x: x > 1, unique_cnts))) / np.size(x)
            avg_comp_no_sing += compr_no_sing

            if np.size(scaled_cnts) < plotrange:
                continue
            avgs = np.add(avgs, scaled_cnts)

        avg_gt_5 /= len(queries)
        avg_comp /= len(queries)
        avg_comp_no_sing /= len(queries)
        print("Average # chains greater than 5: {}".format(avg_gt_5))
        print("Average compression ratio: {}".format(1.0 / avg_comp))
        print("Average compression ratio with singletons removed: {}".format(1.0 / avg_comp_no_sing))
            
        avgs = np.true_divide(avgs, len(queries))
        x = np.array(range(plotrange))
        plt.plot(x, avgs[::-1])

        x_ticks = range(0, plotrange + 1, 20)
        plt.xticks(x_ticks, x_ticks)

        y_ticks = np.round(np.linspace(min(avgs), max(avgs), 5), 5)
        plt.yticks(y_ticks, y_ticks)
        plt.xlabel("Relative chain size (1st largest, 2nd largest, etc.)")
        plt.ylabel("Percentage of total anchors")
        plt.title("Average distribution of chain lengths in: {}".format(fn))
    plt.show()

def singletons_vs_quality(queries):
    singleton_ratios = []
    quality_scores = []
    for query in queries:
        x, y = np.array(query.x), np.array(query.y)
        diffs = np.subtract(x, y)
        unique_cnts = np.unique(diffs, return_counts=True)[1]
        sr = len(list(filter(lambda x: x == 1, unique_cnts))) / np.size(x)
        log_sr = np.log10(1 - sr) #"Proportion that isn't singletons"
        singleton_ratios.append(log_sr) 
        quality_scores.append(query.score)
    
    plt.scatter(quality_scores, singleton_ratios)
    with open("afile.txt", 'w+') as afile:
        for idx, rat in enumerate(singleton_ratios):
            theline = "{},{}\n".format(rat, quality_scores[idx])
            if "inf" not in theline:
                afile.write(theline)
    plt.xlabel("Read quality score (avg over all bases)")
    plt.ylabel("Proportion of non-singleton anchors (log10)")
    plt.title("Proportion of anchors that are non-singleton vs read quality")
    plt.show()


def filter_anchors(query):
    non_singles = BloomFilter(max_elements=1000, error_rate=0.1)
    singles = BloomFilter(max_elements=5800, error_rate=0.1)

    priority0_x, priority1_x, priority2_x = [], [], []
    priority0_y, priority1_y, priority2_y = [], [], []
    for i in range(len(query.x)):
        d = str(((query.x[i] - query.y[i]) >> 4) << 4)
        if d in non_singles:
            priority2_x.append(query.x[i])
            priority2_y.append(query.y[i])
        else:
            non_singles.add(d)
            priority0_x.append(query.x[i])
            priority0_y.append(query.y[i])

        #if d in non_singles:
        #    priority2_x.append(query.x[i])
        #    priority2_y.append(query.y[i])
        #elif d in singles:
        #    non_singles.add(d)
        #    priority1_x.append(query.x[i])
        #    priority1_y.append(query.y[i])
        #else:
        #    singles.add(d)
        #    priority0_x.append(query.x[i])
        #    priority0_y.append(query.y[i])

    #Some stats:
    filter_mem = (non_singles.num_bits_m) / 8192
    original = (128 * len(query.x)) / 8192
    filtered = (128 * len(priority2_x)) / 8192
    print("Original memory use: {}".format(original))
    print("New memory use, data only: {}".format(filtered))
    print("New memory use, with filter: {}".format(filtered + filter_mem))
    print("Savings factor: {}".format(original / (filtered + filter_mem)))
            
    #Priority 0
    #plt.scatter(priority0_x, priority0_y, s=10) 

    #Priority 1
    #plt.scatter(priority1_x, priority1_y, s=10, color='deepskyblue') 

    #Priority 2
    #plt.scatter(priority2_x, priority2_y, s=10, color='r')

    #plt.xlabel("Reference Position")
    #plt.ylabel("Query Position")
    #plt.title("Filtered anchors. Filter size: {}K. Num Hashes: {} False positive Rate: {}".format(filter_mem, 4, 0.1))
    #plt.show()
    return original / (filtered + filter_mem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot anchor output from minimap2. \
            Created by Owen Hoffend.")
    parser.add_argument("fn", help="File with anchor coordinates exported from minimap2")
    parser.add_argument("-n", type=int, default=1, help="Number of queries to read from the file")
    parser.add_argument("-a", action="store_true", help="Read all queries")
    parser.add_argument("--qrange", nargs='+', type=int, help="Query index range, as a python list")
    args = parser.parse_args()
    fn, num_quers, readall = args.fn, args.n, args.a
    q_scores = read_quality(BHAM_DIR)
    queries = load_anchors(fn, num_quers, readall, q_scores) #q_scores as third argument where applicable

    singletons_vs_quality(queries)

    #tot = 0
    #for query in queries:
    #    tot += filter_anchors(query)
    #tot /= len(queries)
    #print("Average bloom filter compression ratio: {}".format(tot))

    #compression_ratio_analysis(fn, queries)
    #merged_queries = Query()
    #merged_queries.merge(queries)
    #plot_chain_lengths(merged_queries)


    #fig, ax = plt.subplots()
    #qrange = args.qrange
    #if qrange != None:
    #    for i in qrange:
    #        q = queries[i-1]
    #        plt.scatter(q.x, q.y, s=10, label=q.name)
    #else:
    #    for q in queries:
    #        plt.scatter(q.x, q.y, s=10, label=q.name)

    #ax.legend(loc='lower left', bbox_to_anchor= (0.0, 1.06), ncol=2,
    #            borderaxespad=0, frameon=False)
    #plt.xlabel("Reference Position")
    #plt.ylabel("Query Position")
    #plt.title("Anchors from: {}".format(fn))
    #plt.show()