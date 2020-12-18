import numpy as np
from scipy.spatial.distance import cdist
from src.Visualizer import Visualizer
from scipy.linalg import lstsq
import pandas as pd
import random
from copy import deepcopy
from tqdm import tqdm
from scipy.special import binom
from GreedyJoining import GreedyJoining

def variant_01(instance, instance_desc, n_planes: int):
    """
    3D Clustering variant 01 for planes that go through the origin. The algorithm works on octants around the origin and
    removes points near the origin as they won't contain much or information at all as all planes go through the origin.
    A octant is selected and a random point in it. Three additional points are selected in the same octant by using the
    minimal transitive distance from the start point. After that a line from the first start point is projected through
    the origin into the octant on the other side and the point with the minimal distance to this line will be selected
    as first end point. Three additional end points are then selected the same way as before. Then the algorithm fits
    a plane through all selected start and end_points and repeats the process until it reaches 8 planes.
    In the second phase the number of planes will be reduced until it matches <n_planes> by eliminating one plane each
    round with the least assigned points. After that all points will be reassigned to the remaining planes. This process
    repeats until there are <n_planes> left.

    :param instance: problem instance containing the points, labels and ideal_planes (the latter is optional as they
        won't be used)
    :param instance_desc: problem description containing the parameters with which the instance was generated
    :param n_planes: int number of plane to find whith the algorithm
    :return: np.array containing the plane equations in the form [a, b, c, d] for ax + by + cz = d
    """
    # TODO: Clean up code
    # TODO: Implement cost function (c, c')
    vis = Visualizer()
    num_all_points = instance_desc["num_points"] * instance_desc["num_planes"]
    # shuffle points and labels respectively based on random permutation
    random_perm = np.random.permutation(range(num_all_points))
    instance["points"] = instance["points"][random_perm, :]
    instance["labels"] = instance["labels"][random_perm]

    # remove every point with distance smaller than x to origin (as they wont contain useful information)
    distance_to_origin = cdist(instance["points"], np.array([[0, 0, 0]]))
    mask = distance_to_origin > .5
    distance_to_origin = distance_to_origin[mask.reshape((num_all_points,))]
    relevant_points = instance["points"][mask.reshape((num_all_points,))]
    relevant_labels = instance["labels"][mask.reshape((num_all_points,))]

    # organize all points in octants from 0 to 7 based on the signs for each coordinate of each point
    # Example: (Positive, Positive, Negative) --> (True, True, False) --> (1,1,0) --> 6 (in binary)
    points_by_octants = []
    for octant in list(range(8)):
        points_by_octants.append(_get_points_in_octant(relevant_points, _get_octant_signs(octant)))
    # get order of octants for execution of algorithm (based on the number of points in them)
    order = np.argsort([len(elem) for elem in points_by_octants])
    # delete entry 0 - 3, because we only need 4 octants. The other 4 will be examined in the process
    order = order[order > 3]

    planes = np.array([])
    for start_octant_idx in order:
        # calculate 1 plane per octant -> range(2) because order only contains 4 of the 8 octants
        for i in range(2):
            # XOR to obtain octant_idx of octant on the other side of the origin
            end_octant_idx = start_octant_idx ^ 7

            if len(points_by_octants[start_octant_idx]) == 0 or len(points_by_octants[end_octant_idx]) == 0:
                continue
            start_points = _get_closest_points_in_octant(random.choice(points_by_octants[start_octant_idx]),
                                                         points_by_octants[start_octant_idx])

            end_distances = []
            for end_point in points_by_octants[end_octant_idx]:
                # distances from all points in destination octant to line from first start_point through origin
                # Idea behind it: We are looking for planes and we assume that the points in the start octant correspond
                # to the same plane. All planes go through the origin and therefore the point with the smallest distance
                # to a line through the origin from a start point in the octant on the other side should be on the plane
                t = -(-np.array(end_point).dot(start_points[0]) / sum(start_points[0] ** 2))
                end_distances = np.append(end_distances, (sum((t * start_points[0] - end_point) ** 2)) ** .5)
            # point with the smallest distance to line through origin and close points near it
            end_points = _get_closest_points_in_octant(points_by_octants[end_octant_idx][np.argsort(end_distances)[0]],
                                                       points_by_octants[end_octant_idx])
            # fit plane through selected points in start and end octant
            fit, _residual, _rnk, _s = _fit_plane(np.append(start_points, end_points, axis=0))

            if planes.shape[0] == 0:
                planes = np.array([[fit[0], fit[1], -1, -fit[2]]])
            else:
                planes = np.append(planes, np.array([[fit[0], fit[1], -1, -fit[2]]]), axis=0)
    # rank each plane and recalculate it based on the labels of every point
    labels = _assign_points_to_plane(instance["points"], planes)
    # recalculate planes and removing the worst plane (in terms of least assigned points)
    for i in range(8):  # TODO: Find a better way which doesn't use a magic number
        planes = _recalculate_planes(planes, instance["points"], labels)
        if planes.shape[0] > n_planes:
            planes = np.delete(planes, pd.Series(labels).value_counts().argmin(), axis=0)
        labels = _assign_points_to_plane(instance["points"], planes)

    # vis.draw_points(instance["points"], _assign_points_to_plane(instance["points"], planes))
    # vis.draw_points(instance["points"], instance["labels"])

    return planes


def _assign_points_to_plane(points, planes):
    """
    Helper function used by variant_01. This function labels every point based on the (minimal) distance from each
    plane.

    :param points: np.array in the shape (n_points, 3) that should be labeled
    :param planes: np.array containing all planes as plane equation in the form [a, b, c, d] for ax + by + cz = d
    :return: np.array of shape (n_points,) containing the labels for the points
    """
    labels = []
    for point in points:
        distances = []
        for plane in planes:
            # distance from every plane the point
            distances.append(abs(plane[:3].dot(point) + plane[3]) / np.linalg.norm(plane[:3]))
        # label is the index of the plane with the minimal distance
        labels.append(np.argmin(distances))
    return np.array(labels)


def _fit_plane(points):
    """
    Helper function used by variant_01 to calculate a plane based on given points. The returned plane will be the best
    fit through all points.

    :param points: np.array of shape (n_points, 3) containing all points
    :return: 4-tuple in the form fit, residual, rnk , s as return values from lstsq from scipy.linalg
    """
    # deepcopy is necessary because the line after it would also change z_axis if its not a copy
    z_axis = deepcopy(points[:, 2])
    points[:, 2] = np.ones(points.shape[0])
    return lstsq(points, z_axis)


def _recalculate_planes(planes, points, labels):
    """
    Helper function used by variant_01. The function fits planes based on the given labels and returns them.

    :param planes: np.array containing all planes as plane equation in the form [a, b, c, d] for ax + by + cz = d
    :param points: np.array in the shape (n_points, 3) containing all points
    :param labels: np.array in the shape (n_labels, ) containing the labels for the planes as Indices of the planes list
    :return: np.array of new calculated planes
    """
    new_planes = []
    for plane_idx, plane in enumerate(planes):
        idxs = np.where(labels == plane_idx)[0]
        fit, residual, rnk, s = _fit_plane(points[idxs])
        # TODO: New plane as a combination from old and new plane maybe better?

        if len(new_planes) == 0:
            new_planes = np.array([[fit[0], fit[1], -1, -fit[2]]])
        else:
            new_planes = np.append(new_planes, np.array([[fit[0], fit[1], -1, -fit[2]]]), axis=0)

    return new_planes


def _get_closest_points_in_octant(center_point: np.array, octant_points, num_points: int = 3,
                                  include_center: bool = True):
    """
    # Helper function used by variant_01. The function uses transitivity to identify <num_points> close points to the
    center_point. Therefore it finds the closest point to the <center_point> and then the closest one to the previously
    found one. It repeats the process <num_points> times in a given octant. Based on the <include_center> parameter it
    returns <num_points> (if False) or <num_points> and the center_point (if True) as np.array

    :param center_point: np.array of shape (3,) which will be the starting point for distance calculation
    :param octant_points: np.array of shape (n_points, 3) containing all points in a specified octant
    :param num_points: int number of points to find based on the minimal transitive distance
    :param include_center: bool whether to include the <center_point> in the return value or not
    :return: np.array containing the found <num_points> points and maybe the <center_point>
    """
    assert num_points < len(octant_points)
    if include_center:
        points = np.array([center_point])
    else:
        points = np.array([])
    tabu_idx = []
    for idx in range(num_points):
        # TODO: Maybe add maximum distance a point can have to the next one to filter points from different planes
        distances = cdist(np.array([points[idx]]), octant_points)[0]
        tabu_idx.append(np.argmin(distances))
        sort_dist = pd.Series(distances)
        sort_dist.drop(tabu_idx, inplace=True)
        point_idx = sort_dist[sort_dist == sorted(sort_dist)[0]].index[0]

        points = np.append(points, np.array([octant_points[point_idx]]), axis=0)
    return points


def _get_points_in_octant(points, octant_signs):
    """
    Helper function for variant_01. The function filters all given points based on the given octant_signs and returns
    all points with the given signs at the coordinates. See _get_octant_signs for details on signs.

    :param points: np.array of shape (n_points, len(octant_signs)) containing all points that should be considered
    :param octant_signs: list or np.array containing the sign for every coordinate of a point
    :return: np.array containing the points with the given octant_signs
    """
    for coord_axis, elem in enumerate(octant_signs):
        if elem:
            # points on coordinate axis should be positive
            points = points[points[:, coord_axis] > 0]
        else:
            # points on coordinate axis should be negative
            points = points[points[:, coord_axis] < 0]
    return points


def _get_octant_signs(octant):
    """
    Helper function for variant_01. The function returns the signs encoded with True and False (True -> +; False -> -)
    for every coordinate of a 3d point. The signs will be used to partition all points according to octants based on
    their signs.

    :param octant: int number of the octant in range [0, 7]
    :return: bool np.array containing signs for every coordinate of a 3d point
    """
    octants = {0: np.array([False, False, False]),
               1: np.array([False, False, True]),
               2: np.array([False, True, False]),
               3: np.array([False, True, True]),
               4: np.array([True, False, False]),
               5: np.array([True, False, True]),
               6: np.array([True, True, False]),
               7: np.array([True, True, True])}
    return octants[octant]


def _cost_01(points, idx01, idx02, idx03):
    # cost for idxs when associated points are all in different clusters
    # TODO: Implement
    return 1.0


def _cost_02(points, idx01, idx02, idx03):
    # cost for idxs when associated points are all in the same cluster
    # TODO: Implement
    return 1.0


def _build_triples(instance):
    triples = np.array([[0, 0, 0]])
    points = instance["points"]
    counter = 0
    tmp = []
    print(f"Generating {int(binom(len(points), 3))} Triples and associated costs")
    for idx01 in tqdm(range(points.shape[0] - 2), "Generating Starting Triples (Outer Loop)"):
        for idx02 in range(idx01 + 1, points.shape[0] - 1):
            for idx03 in range(idx02 + 1, points.shape[0]):
                # build temporary lists and appending them every 20 million values (building lists is faster than
                # np.append but lists need a fuck-ton of RAM)
                counter += 1
                tmp.append([idx01, idx02, idx03])#, _cost_01(points, idx01, idx02, idx03), _cost_02(points, idx01, idx02, idx03)])
                if counter % 20e6 == 0:
                    triples = np.append(triples, np.array(tmp), axis=0)
                    tmp = []
                    counter = 0
    if len(tmp) > 0:
        triples = np.append(triples, np.array(tmp), axis=0)
    return triples[1:]


def _build_index_dict(instance, triples):
    idx_dict = {}
    print(f"Generating Index Dictionary with {len(instance['points'])} Entries")
    for idx in tqdm(range(instance["points"].shape[0]), "Generating Index Dictionary"):
        idx_dict[idx] = np.where(triples == idx)[0]
    return idx_dict


def variant_02(instance, instance_desc):
    # TODO: Implement greedy joining, moving or kernighan and lin
    # Greedy Joining
    triples = _build_triples(instance)
    clusters = np.array(range(instance["points"].shape[0]))
    min_score = 1e100
    previous_min_score = deepcopy(min_score)

    while True:
        print("Calculating scores for joins")
        min_triple_idx = None
        for idx, j_triple in enumerate(triples):
            tmp_score = 0
            tmp_clusters = deepcopy(clusters)

            if len(np.unique([tmp_clusters[int(p)] for p in j_triple[:3]])) == 1:
                # all points are in the same cluster making the join redundant --> skip
                continue

            # TODO: Outsource this block into a separate function
            # merge clusters associated with the points in the triple, as there are at least two clusters
            replace_idxs = []
            for t_elem in j_triple[1:3]:
                replace_idxs += np.where(tmp_clusters == tmp_clusters[int(t_elem)])
            replace_idxs = np.unique(replace_idxs).astype(int)
            # replaces all cluster ids with the cluster id of the first point in the triple
            tmp_clusters[replace_idxs] = tmp_clusters[int(j_triple[0])]

            # calculate score for this clustering with the triple join
            for s_triple in tqdm(triples, "Inner Loop"):
                num_clusters = len(np.unique([tmp_clusters[int(p)] for p in s_triple[:3]]))
                if num_clusters == 1:
                    # points in the same cluster --> use triple idx 4, i.e. cost_02 for points in the same cluster
                    tmp_score += s_triple[4]
                elif num_clusters == 3:
                    # points in different clusters --> use triple idx 3, i.e. cost_01 for points in different clusters
                    tmp_score += s_triple[3]

            if tmp_score < min_score:
                # current score is better --> replace it
                min_score = tmp_score
                min_triple_idx = idx

        if min_score < previous_min_score:
            # TODO: Outsource this block into a separate function
            # new min_score is smaller than previous one, i.e. clustering gets better --> perform join with triple
            replace_idxs = []
            for t_elem in triples[min_triple_idx][1:3]:
                replace_idxs += np.where(clusters == clusters[int(t_elem)])
            replace_idxs = np.unique(replace_idxs).astype(int)
            # replaces all cluster ids with the cluster id of the first point in the triple
            clusters[replace_idxs] = clusters[int(triples[min_triple_idx][0])]
        else:
            # new min_score is higher than previous one, i.e. clustering gets worse --> terminate
            break

    print("Breakpoint")


def variant_03(instance, instance_desc):
    # TODO: Try naive approach with triples
    # Maybe ranking multiple planes will yield good results, similar to variant_01
    # octant approach from variant_01? plane -> number of associated points as ranking?
    # triple -> 2 points + origin and distance to third point for each pair out of them
    # triple -> plane through 3 points and distance to origin
    triples = _build_triples(instance)
    clusters = np.array(range(instance["points"].shape[0]))
    cluster_sizes = np.ones(clusters.shape, dtype=int)
    print("Starting C Implementation")
    joining = GreedyJoining(triples.astype(int), clusters.astype(int), cluster_sizes)
    result = joining.start()

    pass
