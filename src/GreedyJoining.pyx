
import numpy as np
from itertools import combinations
cimport numpy as cnp
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class GreedyJoining:

    cdef int[:,:] triples_view
    cdef int[:] clusters_view
    cdef int[:] cluster_sizes_view

    def __cinit__(self, int[:,:] triples, int[:] clusters, int[:] cluster_sizes):
        self.triples_view = triples
        self.clusters_view = clusters
        self.cluster_sizes_view = cluster_sizes


    def _get_cluster_indices(self, int cluster):
        # Alle Punkt Indices mit der Cluster-ID <cluster> als Liste zurück
        cdef int cluster_a_dim = self.cluster_sizes_view[cluster]
        cdef cnp.ndarray[cnp.int_t, ndim=1] cluster_a = np.zeros(cluster_a_dim, dtype=int)
        cdef int[:] cluster_a_view = cluster_a
        cdef Py_ssize_t x
        cdef int idx_counter = 0

        for x in range(len(self.clusters_view)):
            if self.clusters_view[x] == cluster:
                cluster_a_view[idx_counter] = x
                idx_counter = idx_counter + 1
            if idx_counter == cluster_a_dim:
                # TODO: Maybe remove this, as the cluster_view list doesn't contain many elements and this could slow
                #  down the iteration process
                # Not at the end of the clusters_view but all cluster_idxs were found
                break
        return cluster_a

    def _get_residual_indices(self, int remove_cluster_01, int remove_cluster_02):
        # Alle Punkt Indices ohne Punkte in den Clustern mit ID <remove_cluster_01>, <remove_cluster_02>
        cdef int rem_cluster_dim = self.cluster_sizes_view[remove_cluster_01] + self.cluster_sizes_view[remove_cluster_02]
        cdef int cluster_dim = len(self.clusters_view) - rem_cluster_dim
        cdef cnp.ndarray[cnp.int_t, ndim=1] cluster_a = np.zeros(cluster_dim, dtype=int)
        cdef int[:] cluster_a_view = cluster_a
        cdef Py_ssize_t x
        cdef int idx_counter = 0

        for x in range(len(self.clusters_view)):
            if self.clusters_view[x] != remove_cluster_01 and self.clusters_view[x] != remove_cluster_02:
                cluster_a_view[idx_counter] = x
                idx_counter = idx_counter + 1
            if idx_counter == cluster_dim:
                # TODO: Maybe remove this, as the cluster_view list doesn't contain many elements and this could slow
                #  down the iteration process
                # Not at the end of the clusters_view but all cluster_idxs were found
                break
        return cluster_a

    cdef double _cT(self, int min_elem, int med_elem, int max_elem):
        # TODO: Implement this one
        return 1.

    cdef double _c(self, int min_elem, int med_elem, int max_elem):
        # TODO: Implement this one
        return 1.


    cdef double _schema_aab(self, int[:] cluster_a, int[:] cluster_b, double score):
        # get point indices of points in cluster a, b and define memory views on these
        #cdef cnp.ndarray[cnp.int_t, ndim=1] cluster_a = self._get_cluster_indices(cluster_a_idx)
        #cdef cnp.ndarray[cnp.int_t, ndim=1] cluster_b = self._get_cluster_indices(cluster_b_idx)
        cdef int[:] cluster_a_view = cluster_a
        cdef int[:] cluster_b_view = cluster_b
        cdef Py_ssize_t b_idx
        cdef int a,b,c
        cdef int max_elem, med_elem, min_elem

        # calculate all (a_1, a_2, b_1) combinations from clusters a, b and add the cT score for every combination
        for tuple in combinations(cluster_a_view, 2):
            for b_idx in range(len(cluster_b_view)):
                a = tuple[0]
                b = tuple[1]
                c = cluster_b_view[b_idx]
                # get order of elements
                if a > b:
                    if a > c:
                        max_elem = a
                        if b > c:
                            med_elem = b
                            min_elem = c
                        else:
                            med_elem = c
                            min_elem = b

                    else:
                        med_elem = a
                        max_elem = c
                        min_elem = b
                else:
                    if b > c:
                        max_elem = b
                        if a > c:
                            med_elem = a
                            min_elem = c
                        else:
                            med_elem = c
                            min_elem = a
                    else:
                        med_elem = b
                        max_elem = c
                        min_elem = a
                score = score + self._cT(min_elem, med_elem, max_elem)
        return score


    cdef double _schema_abc(self, int[:] cluster_a, int[:] cluster_b, int[:] cluster_c, double score, bint abc):
        #cdef cnp.ndarray[cnp.int_t, ndim=1] cluster_a = self._get_cluster_indices(cluster_a_idx)
        #cdef cnp.ndarray[cnp.int_t, ndim=1] cluster_b = self._get_cluster_indices(cluster_b_idx)
        #cdef cnp.ndarray[cnp.int_t, ndim=1] cluster_c
        #if abc:
        #    cluster_c = self._get_cluster_indices(cluster_c_idx)
        #else:
        #    cluster_c = self._get_residual_indices(cluster_a_idx, cluster_b_idx)
        cdef int[:] cluster_a_view = cluster_a
        cdef int[:] cluster_b_view = cluster_b
        cdef int[:] cluster_c_view = cluster_c
        cdef Py_ssize_t a_idx, b_idx, c_idx
        cdef int a,b,c
        cdef int max_elem, med_elem, min_elem

        for a_idx in range(len(cluster_a_view)):
            for b_idx in range(len(cluster_b_view)):
                for c_idx in range(len(cluster_c_view)):
                    a = cluster_a_view[a_idx]
                    b = cluster_b_view[b_idx]
                    c = cluster_c_view[c_idx]
                    # get order of elements
                    if a > b:
                        if a > c:
                            max_elem = a
                            if b > c:
                                med_elem = b
                                min_elem = c
                            else:
                                med_elem = c
                                min_elem = b

                        else:
                            med_elem = a
                            max_elem = c
                            min_elem = b
                    else:
                        if b > c:
                            max_elem = b
                            if a > c:
                                med_elem = a
                                min_elem = c
                            else:
                                med_elem = c
                                min_elem = a
                        else:
                            med_elem = b
                            max_elem = c
                            min_elem = a

                    # calculate the -c cost
                    score = score - self._c(min_elem, med_elem, max_elem)
                    if abc:
                        # This is the abc case --> + cT cost again
                        score = score + self._cT(min_elem, med_elem, max_elem)


    def start(self):
        return self.join()

    def join(self):
        cdef double min_score = 1e100
        cdef previous_min_score = 1e101
        cdef int min_triple_idx
        cdef double tmp_score
        cdef int [:] tmp_clusters = np.zeros(len(self.clusters_view), dtype=int)
        cdef int idx
        cdef cnp.ndarray[cnp.int_t, ndim=1] different_clusters = np.zeros(3, dtype=int)
        cdef cnp.ndarray[cnp.int_t, ndim=1] unique_clusters
        cdef Py_ssize_t j_triple_idx
        cdef Py_ssize_t j_triple_elem_idx
        #cdef Py_ssize_t tmp_elem_idx
        #cdef Py_ssize_t tmp_cluster_idx
        cdef int[:] j_triple
        cdef int[:] cluster_a
        cdef int[:] cluster_b
        cdef int[:] cluster_c
        cdef int[:] cluster_y01
        cdef int[:] cluster_y02
        cdef int[:] cluster_y12
        #cdef list replace_idxs_list
        #cdef int replace_cluster_id
        #cdef int current_cluster_id

        while True:
            # Calculating scores for joins
            # TODO: Change iteration from iterating over triples to iterating over clusters -> min 1% performance gain per iteration
            idx = 0

            for j_triple_idx in range(len(self.triples_view)):
                j_triple = self.triples_view[j_triple_idx]
                tmp_score = min_score
                tmp_clusters[...] = self.clusters_view

                if j_triple_idx % 1e6 == 0:
                    print(f"Triple {j_triple_idx}")
                for j_triple_elem_idx in range(3):
                    different_clusters[j_triple_elem_idx] = tmp_clusters[j_triple_elem_idx]
                unique_clusters = np.unique(different_clusters)
                if len(unique_clusters) == 1:
                    # all points are in the same cluster making the join redundant --> skip
                    continue
                elif len(unique_clusters) == 2:
                    # two different clusters
                    cluster_a = self._get_cluster_indices(unique_clusters[0])
                    cluster_b = self._get_cluster_indices(unique_clusters[1])
                    cluster_y01 = self._get_residual_indices(unique_clusters[0], unique_clusters[1])

                    # all (a_1, a_2, b_1) and (b_1, b_2, a_1) triples will change
                    tmp_score = self._schema_aab(cluster_a, cluster_b, tmp_score)
                    tmp_score = self._schema_aab(cluster_b, cluster_a, tmp_score)
                    # all (a_1, b_1, y_1) triples will change (y - all nodes not in a or b)
                    tmp_score = self._schema_abc(cluster_a, cluster_b, cluster_y01, tmp_score, False)
                elif len(unique_clusters) == 3:
                    # three different clusters
                    cluster_a = self._get_cluster_indices(unique_clusters[0])
                    cluster_b = self._get_cluster_indices(unique_clusters[1])
                    cluster_c = self._get_cluster_indices(unique_clusters[2])
                    cluster_y01 = self._get_residual_indices(unique_clusters[0], unique_clusters[1])
                    cluster_y02 = self._get_residual_indices(unique_clusters[0], unique_clusters[2])
                    cluster_y12 = self._get_residual_indices(unique_clusters[1], unique_clusters[2])

                    # all (a_1, b_1, c_1) triple scores will be swapped (c -> cT)
                    tmp_score = self._schema_abc(cluster_a, cluster_b, cluster_c, tmp_score, True)
                    # all (a_1, a_2, b_1), (b_1, b_2, a_1), (b_1, b_2, c_1), (c_1, c_2, b_1), (a_1, a_2, c_1), (c_1, c_2, a_1)
                    # triple scores will change
                    tmp_score = self._schema_aab(cluster_a, cluster_b, tmp_score)
                    tmp_score = self._schema_aab(cluster_b, cluster_a, tmp_score)
                    tmp_score = self._schema_aab(cluster_b, cluster_c, tmp_score)
                    tmp_score = self._schema_aab(cluster_c, cluster_b, tmp_score)
                    tmp_score = self._schema_aab(cluster_a, cluster_c, tmp_score)
                    tmp_score = self._schema_aab(cluster_c, cluster_a, tmp_score)
                    # all (a_1, b_1, y_1), (a_1, c_1, y_1), (b_1, c_1, y_1) triples will change (y - all nodes not in
                    # a or b)
                    tmp_score = self._schema_abc(cluster_a, cluster_b, cluster_y01, tmp_score, False)
                    tmp_score = self._schema_abc(cluster_a, cluster_c, cluster_y02, tmp_score, False)
                    tmp_score = self._schema_abc(cluster_b, cluster_c, cluster_y12, tmp_score, False)
                else:
                    print("Unknown length of unique_clusters array in join function in GreedyJoining.pyx")

                if tmp_score < min_score:
                    # current score is better --> replace it
                    min_score = tmp_score
                    min_triple_idx = idx

                idx += 1

            print("Iteration done")
            if min_score < previous_min_score:
                # new min_score is smaller than previous one, i.e. clustering gets better --> perform join with triple
                replace_idxs_list = []
                replace_cluster_id = self.clusters_view[j_triple[0]]
                replace_cluster_size = self.cluster_sizes_view[replace_cluster_id]
                for tmp_elem_idx in range(2):
                    # add size of cluster at idx 1 or 2 (based on tmp_elem_idx) to cluster size at idx 0 as they get merged
                    current_cluster_id = self.clusters_view[j_triple[tmp_elem_idx + 1]]
                    for cluster_idx in range(len(self.clusters_view)):
                        if self.clusters_view[cluster_idx] == current_cluster_id:
                            replace_idxs_list.append(cluster_idx)

                # TODO: Maybe make a for loop out of this line
                self.clusters_view[np.array(replace_idxs_list, dtype=int)] = replace_cluster_id
                self.cluster_sizes_view[replace_cluster_id] += len(replace_idxs_list)
            else:
                # new min_score is higher than previous one, i.e. clustering gets worse --> terminate
                break
        return np.array(self.clusters_view)








