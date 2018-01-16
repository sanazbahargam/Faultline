import sys
import random
import numpy as np
import app_faultline as fl
from munkres import Munkres
from operator import itemgetter
from collections import defaultdict
from multiprocessing import Pool


def compute_centers(data, clusters):
    p_num = len(data)
    f_num = len(data[0])
    g_num = p_num / len(clusters)
    centers = []
    centers_choose2 = []
    centers_cost = []
    for c in clusters:
        center = [defaultdict(int) for f in xrange(f_num)]
        for ind in c:
            for f in xrange(f_num):
                center[f][data[ind][f]] = center[f][data[ind][f]] + 1
        centers.append(center)
        # computing the choose2's
        choose2 = 0
        cost = 0
        for f in xrange(f_num):
            for v in center[f].values():
                choose2 = choose2 + (v * (v - 1) / 2)
                cost = cost + (v * (v - 1) / 2) * (g_num - v)
        centers_choose2.append(choose2)
        centers_cost.append(cost)
    return centers, centers_choose2, centers_cost


# Evaluates distance of a point to a cluster center (for hybrid method)
def dist_hybrid(p, center, choose2, g_num):
    val = choose2
    for p_f, f_vals in zip(p, center):
        val = val - (f_vals[p_f] * (f_vals[p_f] - 1) / 2)
        val = val + (f_vals[p_f] * (g_num - f_vals[p_f]))
    return val


# Evaluates distance of a point to a cluster center (for hybrid method)
def dist_hybrid_full(p, center, choose2, cost, g_num):
    val = choose2 + cost
    for p_f, f_vals in zip(p, center):
        val = val - (f_vals[p_f] * (f_vals[p_f] - 1) / 2)
        val = val + (f_vals[p_f] * (g_num - f_vals[p_f]))
    return val


# Evaluates distance of a point to a cluster center (for homogeneous method)
def dist_homo(p, center):
    val = 0
    for f in range(len(center)):
        f_vals = center[f]
        for k in f_vals.keys():
            # counting dissimilarities
            if k != p[f]:
                val = val + f_vals[k]
    return val


# Evaluates distance of a point to a cluster center (for homogeneous method)
def dist_splitter(p, center, choose2, cost, g_num, is_same, n1, n2):
    if is_same:
        val = cost / float(n1)
    else:
        val = cost + choose2
        for p_f, f_vals in zip(p, center):
            val = val - (f_vals[p_f] * (f_vals[p_f] - 1) / 2)
            val = val + (f_vals[p_f] * (g_num - f_vals[p_f]))
        val = val / float(n2)
    return val


def dist_tri_homo(p, center):
    val = 0
    for f in range(len(center)):
        f_vals = center[f]
        for k in f_vals.keys():
            if k != p[f]:
                val = val + (f_vals[k] * (f_vals[k] - 1) / 2)
    return val


def dist_tri_homo_full(p, center, g_num):
    val = ((g_num + 1) * g_num * (g_num - 1) / 6)
    for f in range(len(center)):
        f_vals = center[f]
        for k in f_vals.keys():
            if k != p[f]:
                val = val - (f_vals[k] * (f_vals[k] - 1) * (f_vals[k] - 2) / 6)
            else:
                val = val - ((f_vals[k] + 1) * f_vals[k] * (f_vals[k] - 1) / 6)
    return val


def dist_tri_homo_norm(p, center, choose2, cost, g_num, is_same, n1, n2):
    if not is_same:
        val = ((g_num + 1) * g_num * (g_num - 1) / 6)
        for f in range(len(center)):
            f_vals = center[f]
            for k in f_vals.keys():
                if k != p[f]:
                    val = val - (f_vals[k] * (f_vals[k] - 1) *
                                 (f_vals[k] - 2) / 6)
                else:
                    val = val - ((f_vals[k] + 1) * f_vals[k] *
                                 (f_vals[k] - 1) / 6)
        val = val / float(n2)
    else:
        val = (g_num * (g_num - 1) * (g_num - 2) / 6)
        for f in range(len(center)):
            f_vals = center[f]
            for k in f_vals.keys():
                val = val - (f_vals[k] * (f_vals[k] - 1) * (f_vals[k] - 2) / 6)
        val = val / float(n1)
    return val


def dist_tri_hetro_norm(p, center, choose2, cost, g_num, is_same, n1, n2):
    if not is_same:
        val = 0
        for f in range(len(center)):
            f_vals = center[f]
            for k in f_vals.keys():
                if k == p[f]:
                    val = val + (f_vals[k] * (g_num - f_vals[k]))
                val = val + (f_vals[k] * (f_vals[k] - 1) / 2)
        val = val / float(n2)
    else:
        val = 0
        for f in range(len(center)):
            f_vals = center[f]
            for k in f_vals.keys():
                if k == p[f]:
                    val = val + ((f_vals[k] - 1) * (g_num - f_vals[k] + 1))
                val = val + (f_vals[k] * (f_vals[k] - 1) / 2)
        val = val / float(n1)
    return val


def dist_tri_hetro(p, center, g_num):
    val = 0
    for f in range(len(center)):
        f_vals = center[f]
        for k in f_vals.keys():
            if k == p[f]:
                val = val + (f_vals[k] * (g_num - f_vals[k]))
            val = val + (f_vals[k] * (f_vals[k] - 1) / 2)
    return val


# Evaluates distance of a point to a cluster center (for hetrogeneous method)
def dist_hetro(p, center):
    val = 0
    for f in range(len(center)):
        f_vals = center[f]
        for k in f_vals.keys():
            if k == p[f]:
                val = val + f_vals[k]
    return val


# Hungarian method to guarantee a balanced assignment of points
def munk_iter(data, clusters, centers, centers_choose2, centers_cost, algo):
    p_num = len(data)
    g_num = p_num / len(centers)
    # computing the distance matrix
    matrix = [None] * p_num
    for i in xrange(p_num):
        dist = [0] * p_num
        for j in xrange(len(centers)):
            if algo == 'homo':
                d_temp = dist_homo(data[i], centers[j])
            elif algo == 'hetro':
                d_temp = dist_hetro(data[i], centers[j])
            elif algo == 'tri_homo':
                d_temp = dist_tri_homo(data[i], centers[j])
            elif algo == 'tri_hetro':
                d_temp = dist_tri_hetro(data[i], centers[j], g_num)
            elif algo == 'tri_homo_full':
                d_temp = dist_tri_homo_full(data[i], centers[j], g_num)
            elif algo == 'hybrid':
                d_temp = dist_hybrid(data[i], centers[j],
                                     centers_choose2[j], g_num)
            elif algo == 'hybrid_full':
                d_temp = dist_hybrid_full(data[i], centers[j],
                                          centers_choose2[j],
                                          centers_cost[j], g_num)
            else:
                is_same = (i in clusters[j])
                part1 = g_num / 2
                part2 = g_num - part1
                n1 = float((part1 * (part1 - 1) / 2 * part2) +
                           (part2 * (part2 - 1) / 2 * part1))
                part1 += 1
                n2 = float((part1 * (part1 - 1) / 2 * part2) +
                           (part2 * (part2 - 1) / 2 * part1))
                if algo == 'splitter_hybrid':
                    d_temp = dist_splitter(data[i], centers[j],
                                           centers_choose2[j], centers_cost[j],
                                           g_num, is_same, n1, n2)
                else:
                    d_temp = dist_tri_homo_norm(data[i], centers[j],
                                                centers_choose2[j],
                                                centers_cost[j], g_num,
                                                is_same, n1, n2)
            for t in xrange(g_num):
                dist[j * g_num + t] = d_temp
        matrix[i] = dist
    # doing the Hungarian assignment
    m = Munkres()
    indexes = m.compute(matrix)
    # reading the assignment and redoing the clsutering
    new_clusters = [[] for x in xrange(len(clusters))]
    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total = total + value
        new_clusters[column / g_num].append(row)
    return new_clusters


# Performs a single iteration of the k-means like algorithm
def greedy_iter(data, clusters, centers, centers_choose2, centers_cost, algo):
    p_num = len(data)
    g_num = p_num / len(centers)
    list_p = [0]*(p_num*len(clusters))
    list_k = [0]*(p_num*len(clusters))
    list_d = [0]*(p_num*len(clusters))
    # computing the distance matrix
    t = 0
    for i in xrange(len(data)):
        for j in xrange(len(centers)):
            if algo == 'homo':
                d_temp = dist_homo(data[i], centers[j])
            elif algo == 'hetro':
                d_temp = dist_hetro(data[i], centers[j])
            elif algo == 'tri_homo':
                d_temp = dist_tri_homo(data[i], centers[j])
            elif algo == 'tri_hetro':
                d_temp = dist_tri_hetro(data[i], centers[j], g_num)
            elif algo == 'tri_homo_full':
                d_temp = dist_tri_homo_full(data[i], centers[j], g_num)
            elif algo == 'hybrid':
                d_temp = dist_hybrid(data[i], centers[j],
                                     centers_choose2[j], g_num)
            elif algo == 'hybrid_full':
                d_temp = dist_hybrid_full(data[i], centers[j],
                                          centers_choose2[j],
                                          centers_cost[j], g_num)
            else:
                is_same = (i in clusters[j])
                part1 = g_num / 2
                part2 = g_num - part1
                n1 = float((part1 * (part1 - 1) / 2 * part2) +
                           (part2 * (part2 - 1) / 2 * part1))
                part1 += 1
                n2 = float((part1 * (part1 - 1) / 2 * part2) +
                           (part2 * (part2 - 1) / 2 * part1))
                if algo == 'splitter_hybrid':
                    d_temp = dist_splitter(data[i], centers[j],
                                           centers_choose2[j], centers_cost[j],
                                           g_num, is_same, n1, n2)
                else:
                    d_temp = dist_tri_homo_norm(data[i], centers[j],
                                                centers_choose2[j],
                                                centers_cost[j], g_num,
                                                is_same, n1, n2)
            list_d[t] = d_temp
            list_p[t] = i
            list_k[t] = j
            t = t + 1
    edges = sorted(zip(list_d, zip(list_p, list_k)))
    # doing the greedy min-weight assignment
    node_stat = [False]*p_num
    cluster_stat = [g_num]*len(centers)
    new_clusters = [[] for x in xrange(len(clusters))]
    total = 0
    for (d, (i, j)) in edges:
        if node_stat[i]:
            continue
        if cluster_stat[j] == 0:
            continue
        # updating the clusters
        new_clusters[j].append(i)
        node_stat[i] = True
        cluster_stat[j] = cluster_stat[j] - 1
        total = total + d
    return new_clusters


def fast_greedy_iter(data, clusters, centers, centers_choose2,
                     centers_cost, algo, p):
    p_num = len(data)
    g_num = p_num / len(centers)
    list_p = [0]*(p_num*len(clusters))
    list_k = [0]*(p_num*len(clusters))
    list_d = [0]*(p_num*len(clusters))
    params = [[] for x in xrange(p_num*len(clusters))]
    # creating the param_list
    t = 0
    for i in xrange(p_num):
        for j in xrange(len(clusters)):
            params[t].append(data[i])
            params[t].append(centers[j])
            if algo != 'homo' and algo != 'hetro' and algo != 'tri_homo':
                if algo == 'tri_hetro' or algo == 'tri_homo_full':
                    params[t].append(g_num)
                else:
                    params[t].append(centers_choose2[j])
                    if algo == 'hybrid':
                        params[t].append(g_num)
                    else:
                        params[t].append(centers_cost[j])
                        params[t].append(g_num)
                        if algo != 'hybrid_full':
                            is_same = (i in clusters[j])
                            part1 = g_num / 2
                            part2 = g_num - part1
                            n1 = float((part1 * (part1 - 1) / 2 * part2) +
                                       (part2 * (part2 - 1) / 2 * part1))
                            part1 += 1
                            n2 = float((part1 * (part1 - 1) / 2 * part2) +
                                       (part2 * (part2 - 1) / 2 * part1))
                            params[t].append(is_same)
                            params[t].append(n1)
                            params[t].append(n2)
            list_p[t] = i
            list_k[t] = j
            t = t + 1
    # computing the distance matrix (in parallel)
    if algo == 'homo':
        list_d = p.map(dist_homo_unwrap, params, len(params)/12)
    elif algo == 'hetro':
        list_d = p.map(dist_hetro_unwrap, params, len(params)/12)
    elif algo == 'tri_homo':
        list_d = p.map(dist_tri_homo_unwrap, params, len(params)/12)
    elif algo == 'tri_hetro':
        list_d = p.map(dist_tri_hetro_unwrap, params, len(params)/12)
    elif algo == 'tri_homo_full':
        list_d = p.map(dist_tri_homo_full_unwrap, params, len(params)/12)
    elif algo == 'hybrid':
        list_d = p.map(dist_hybrid_unwrap, params, len(params)/12)
    elif algo == 'hybrid_full':
        list_d = p.map(dist_hybrid_full_unwrap, params, len(params)/12)
    elif algo == 'tri_homo_norm':
        list_d = p.map(dist_tri_homo_norm_unwrap, params, len(params)/12)
    elif algo == 'tri_hetro_norm':
        list_d = p.map(dist_tri_hetro_norm_unwrap, params, len(params)/12)
    else:
        list_d = p.map(dist_splitter_unwrap, params, len(params)/12)

    edges = sorted(zip(list_d, zip(list_p, list_k)))

    # doing the greedy min-weight assignment
    node_stat = [False]*p_num
    cluster_stat = [g_num]*len(centers)
    new_clusters = [[] for x in xrange(len(clusters))]
    total = 0
    for (d, (i, j)) in edges:
        # checking if point i is not selecsplitter
        if node_stat[i]:
            continue
        # checking if cluster j has room
        if cluster_stat[j] == 0:
            continue
        # updating the clusters
        new_clusters[j].append(i)
        node_stat[i] = True
        cluster_stat[j] = cluster_stat[j] - 1
        total = total + d
    return new_clusters


def dist_hybrid_unwrap(x):
    return dist_hybrid(*x)


def dist_hybrid_full_unwrap(x):
    return dist_hybrid_full(*x)


def dist_tri_hetro_unwrap(x):
    return dist_tri_hetro(*x)


def dist_tri_homo_unwrap(x):
    return dist_tri_homo(*x)


def dist_hetro_unwrap(x):
    return dist_hetro(*x)


def dist_tri_homo_full_unwrap(x):
    return dist_tri_homo_full(*x)


def dist_homo_unwrap(x):
    return dist_homo(*x)


def dist_splitter_unwrap(x):
    return dist_splitter(*x)


def dist_tri_homo_norm_unwrap(x):
    return dist_tri_homo_norm(*x)


def dist_tri_hetro_norm_unwrap(x):
    return dist_tri_hetro_norm(*x)


# Main routine that performs the K-means like balanced partitioning which
def hung_partition(data, g_num, algo, match_algo, fast, p):
    # check if balanced partitioning is possible
    remainder = len(data) % g_num
    if remainder != 0:
        data = data[0:(len(data) - remainder)]

    p_num = len(data)
    k = p_num / g_num
    # initalizing (creating a random partitioning)
    clusters = []
    for i in range(k):
        cluster = range(i*g_num, (i+1)*g_num)
        clusters.append(cluster)
    # performing 20 iterations
    prev_total = -1
    prevprev_total = -1
    best_val = sys.maxint
    best_cluster = None
    for i in range(20):
        # evaluating the clusters
        total = 0
        for c in clusters:
            data_part = itemgetter(*c)(data)
            total = total + fl.fast_eval(data_part)
        if total < best_val:
            best_val = total
            best_cluster = clusters
        if total == prev_total and total == prevprev_total:
            break
        else:
            prevprev_total = prev_total
            prev_total = total
        # performing the iteration
        centers, centers_choose2, centers_cost = compute_centers(data, clusters)
        if match_algo == 'munk':
            clusters = munk_iter(data, clusters, centers,
                                 centers_choose2, centers_cost, algo)
        else:
            if fast:
                clusters = fast_greedy_iter(data, clusters, centers,
                                            centers_choose2, centers_cost,
                                            algo, p)
            else:
                clusters = greedy_iter(data, clusters, centers,
                                       centers_choose2, centers_cost, algo)
    return best_cluster


# Randomly partitions the individuals
def rand_partition(data, k):
    # check if balanced partitioning is possible
    remainder = len(data) % k
    if remainder != 0:
        data = data[0:len(data) - remainder]
    p_num = len(data)
    g_num = p_num / k
    # initalizing (creating a random partitioning)
    clusters = []
    for i in range(k):
        cluster = range(i*g_num, (i+1)*g_num)
        clusters.append(cluster)
    return clusters


# greedy_partition
def greedy_partition(data, k):
    # check if balanced partitioning is possible
    remainder = len(data) % k
    if remainder != 0:
        data = data[0:len(data) - remainder]

    f_num = len(data[0])
    p_num = len(data)
    g_num = p_num / k
    # initalizing (creating a random partitioning)
    clusters = []
    ids = range(len(data))
    for g in xrange(k):
        # randomly selecting two points
        team = random.sample(ids, 2)
        center = [defaultdict(int) for f in xrange(f_num)]
        # adding the points to the center
        for w in team:
            ids.remove(w)
            for f in xrange(f_num):
                center[f][data[w][f]] += 1
        for i in range(g_num - 2):
            min_tri = sys.maxint
            best_w = -1
            for w in ids:
                point = data[w]
                # evaluate how many bad triangles it creates
                bad = 0
                for f in xrange(f_num):
                    for k, v in center[f].iteritems():
                        if k == point[f]:
                            bad += v * (len(team) - v)
                        else:
                            bad += (v * (v - 1) / 2)
                if bad < min_tri:
                    min_tri = bad
                    best_w = w
            # adding the discovered point
            team.append(best_w)
            # updating the center
            for f in xrange(f_num):
                center[f][data[best_w][f]] += 1
            ids.remove(best_w)
        # putting the team in the clusters
        clusters.append(team)
    return clusters


def evaluate_partition(data, clusters):
    simple_obj = 0             # counts the number of problematic triangles
    for c in clusters:
        data_part = itemgetter(*c)(data)
        simple_obj = simple_obj + fl.fast_eval(data_part)
    return simple_obj


def tri_freq(data):
    all_neg = 0.0
    all_pos = 0.0
    total_cnt = 0.0
    list_good = []
    for i in xrange(len(data)):
        for j in xrange(len(data)):
            for k in xrange(len(data)):
                if i <= j or j <= k or i <= k:
                    continue
                good = 0.0
                for f in xrange(len(data[0])):
                    total_cnt = total_cnt + 1
                    val_i = data[i][f]
                    val_j = data[j][f]
                    val_k = data[k][f]
                    if val_i != val_j and val_j != val_k and val_i != val_k:
                        good = good + 1
                        all_neg = all_neg + 1
                    elif val_i == val_j and val_j == val_k:
                        good = good + 1
                        all_pos = all_pos + 1
                good = good / len(data[0])
                list_good.append(good)
    return(all_neg / total_cnt, all_pos / total_cnt, np.std(list_good))


def make_data(group_num, feature_num, noise, min_size, max_size):
    all_data = []
    group_sizes = []
    for i in range(group_num):
        group_sizes.append(random.randint(min_size, max_size))
    noise_count = 0
    for i in range(len(group_sizes)):
        for j in range(group_sizes[i]):
            person = [0] * feature_num
            for f in range(feature_num):
                person[f] = i
                # adding the noise
                if random.random() < noise:
                    noise_count = noise_count + 1
                    person[f] = random.randint(0, len(group_sizes) - 1)
            all_data.append(person)
    return all_data


def main():
    p = Pool(2)
    # creating toy data
    data = make_data(4, 10, .3, 20, 20)
    num_teams = 5
    team_size = len(data) / num_teams
    for method in ['splitter', 'greedy', 'rand', 'clustering']:
        if method == 'splitter':
            clusters = hung_partition(data, team_size, 'splitter_hybrid',
                                      'greedy', True, p)
        elif method == 'greedy':
            clusters = greedy_partition(data, num_teams)
        elif method == 'rand':
            clusters = rand_partition(data, num_teams)
        elif method == 'clustering':
            clusters = hung_partition(data, team_size, 'homo',
                                      'greedy', True, p)
        print(clusters)


if __name__ == "__main__":
    main()
