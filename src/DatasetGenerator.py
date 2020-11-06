import numpy as np
import random
import pickle
import os


class DatasetGenerator:

    def __init__(self, num_instances: int, num_planes: int, num_points: int, max_deviation: float, seed=42,
                 through_origin: bool = True):
        '''
        The DatasetGenerator can generate problem instances containing approximated 2d planes through randomly sampled
        points in 3D. All points are generated randomly with a given deviation from the ideal generated planes.
        Additionally it can be specified whether the planes should intersect with the origin or not.
        To generate a dataset use the generate_dataset function after generating the object.

        :param num_instances: The number of problem instances in the dataset
        :param num_planes: The number of planes in each problem instance
        :param num_points: The number of points belonging to each plane in each poblem instance
        :param max_deviation: The maximum distance a to a plane associated point can have from the ideal plane
        :param seed: The random seed used in the generation of the problem instances
        :param through_origin: Whether the (ideal) planes should intersect with the origin (0,0,0) or not
        '''
        self._num_instances = num_instances
        self._num_planes = num_planes
        self._num_points = num_points
        self._max_deviation = max_deviation
        self._seed = random.seed() if seed is None else seed
        self._through_origin = through_origin
        # setting seeds
        np.random.seed(self._seed)
        random.seed(self._seed)

    def _generate_instance(self):
        '''
        Internal function of the DatasetGenerator which generates a single problem instances based on the given
        parameters of the DatasetGenerator object
        '''
        instance = {}

        # calculating <num_planes> plane equations and storing them in planes
        planes = []
        for i in range(self._num_planes):
            p2 = np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
            p3 = np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
            if self._through_origin:
                cp = np.cross(p3, p2)
                planes.append(np.array([cp[0], cp[1], cp[2], 0]))
            else:
                p1 = np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
                cp = np.cross(p3 - p1, p2 - p1)
                planes.append(np.array([cp[0], cp[1], cp[2], np.dot(cp, p3)]))

        points = []
        labels = []
        point_label = 0
        # generate <num_points> points belonging to generated planes with <max_deviation> as maximum distance from plane
        for plane in planes:
            # normal vector of plane (scaled to length 1)
            normal_vector = plane[:3] / np.linalg.norm(plane[:3])
            for i in range(self._num_points):
                point = None
                while True:
                    # creating random point and placing it on the ideal plane
                    point = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
                    point = np.append(point, ((plane[0] * point[0] + plane[1] * point[1] - plane[3]) / (-plane[2])))
                    # adding some noise (deviation from ideal plane based on given <max_deviation>)
                    if random.uniform(0, 1) < .5:
                        point -= normal_vector * random.uniform(0, self._max_deviation)
                    else:
                        point += normal_vector * random.uniform(0, self._max_deviation)
                    if not np.any(point > 1) and not np.any(point < -1):
                        break
                points.append(point)
                labels.append(point_label)
            point_label += 1
        instance["ideal_planes"] = np.array(planes)
        instance["points"] = np.array(points)
        instance["labels"] = np.array(labels)
        return instance

    def generate_dataset(self):
        """
        Generates a dataset consisting of <num_instances> different problem instances. Each one contains <num_planes>
        different planes, where each plane has <num_points> associated points which are randomly sampled in 8 unit
        cubes around the origin with a maximum distance <max_deviation> from the generated ideal plane.
        Create an object of the DatasetGenerator before you use this function.

        Returns Tuple <List<Dict>, Dict> in the form <dataset, description> where dataset is the list of instances and
        description is a dictionary with all parameter values in it. The dataset List contains the instances as
        dictionaries with 3 entries ("points" contains a list of 3d points sampled around the planes; "labels"
        contains a list with the labels for the points and "ideal_planes" contains the parameters for the plane
        equations of the ideal planes in the form [a, b, c, d] for ax + by + cz = d)
        CAUTION: If you rearrange the points list, rearrange the labels list with the same permutation as otherwise the
        labels wont match anymore.
        """
        dataset = []
        description = {"num_instances": self._num_instances,
                       "num_planes": self._num_planes,
                       "num_points": self._num_points,
                       "max_deviation": self._max_deviation,
                       "seed": self._seed,
                       "through_origin": self._through_origin}
        for i in range(self._num_instances):
            dataset.append(self._generate_instance())
        return dataset, description

    def generate_and_save_dataset(self, folder: str):
        """
        Generates a dataset based on the given parameters and saves it with the given filename.
        Create an object of the DatasetGenerator before you use this function.

        :param folder: Folder in which the dataset should be stored
        """
        # just in case the user forgot to add a / to the end of his path
        if folder[-1] != "/":
            folder += "/"

        dataset, description = self.generate_dataset()

        # save every problem instance
        for instance_idx in range(len(dataset)):
            with open(folder + f"instance_{instance_idx}.pkl", "wb") as file:
                pickle.dump(dataset[instance_idx], file, protocol=pickle.HIGHEST_PROTOCOL)

        # save the dataset description
        with open(folder + "description.pkl", "wb") as file:
            pickle.dump(description, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_dataset(self, folder: str):
        """
        Loads a previously saved dataset from a given folder.

        :param folder: The folder containing the previously saved instance files and description file.
        :return: Tuple of length 2, the first element is the dataset and the second one is the description of the
            dataset
        """

        # just in case the user forgot to add a / to the end of his path
        if folder[-1] != "/":
            folder += "/"

        files = os.listdir(folder)
        description_file = [name for name in files if name.__contains__("description")]
        instance_files = sorted([name for name in files if name.__contains__("instance")])

        # make sure only one description file exists
        assert len(description_file) == 1

        description = None
        description_file = description_file[0]
        with open(folder + description_file, "rb") as file:
            description = pickle.load(file)

        dataset = []
        for instance_file in instance_files:
            with open(folder + instance_file, "rb") as file:
                dataset.append(pickle.load(file))

        # test if everything is alright (tests the easy parameters for every instance)
        assert len(dataset) == description["num_instances"]
        for instance in dataset:
            assert len(instance["ideal_planes"]) == description["num_planes"]
            assert len(instance["points"]) == description["num_planes"] * description["num_points"]
            assert len(instance["labels"]) == description["num_planes"] * description["num_points"]
            if description["through_origin"]:
                assert sum(dataset[0]['ideal_planes'][:, 3]) == 0
            else:
                assert sum(dataset[0]['ideal_planes'][:, 3]) != 0

        return dataset, description
