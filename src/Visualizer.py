import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Visualizer:

    def __init__(self):
        """
        Visualizer object to draw points and planes. See draw_points for usage.
        """
        self.colors = ["#2F4F4F", "#228B22", "#800000", "#FFFF00", "#0000CD", "#00FF00", "#00FFFF", "#1E90FF",
                       "#FFDEAD", "#FF69B4"]

    def draw_points(self, points: np.array, labels: np.array = None, ideal_planes: np.array = None):
        """
        Draws the given points, either in the same color if no labels are given, or in different colors if they are
        given, and draws the ideal layers semi-transparently (if given).

        :param points: Numpy array in the shape (number_of_points, 3) where each row represents a point in R3
        :param labels: Numpy array in the shape (number_of_points,) containing the labels for the points as ints as
            range from zero to the number of different labels
        :param ideal_planes: Numpy array in the shape (number_of_unique_labels i.e. number_of_planes, 4) where each row
            contains the parameters of a plane equation in the form [a, b, c, d] for ax + by + cz = d
        """
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # drawing the points based on given labels or just all in the same color
        if labels is not None:
            # make sure every point got a label and enough colors are available to draw every plane in a different color
            assert len(points) == len(labels)
            if len(np.unique(labels)) > len(self.colors):
                raise ValueError(f"Cant display more than {len(self.colors)} different planes with different colors.")

            for point_label in range(len(np.unique(labels))):
                # get indices of all points with the same label and then draw them with the same color
                points_to_draw = points[np.where(labels == point_label)[0]]
                ax.scatter(points_to_draw[:, 0], points_to_draw[:, 1], points_to_draw[:, 2], marker="o",
                           c=self.colors[point_label])
        else:
            # no labels, just draw every point in the same color
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker="o", c=self.colors[0])

        # drawing the ideal planes
        if ideal_planes is not None:
            for plane in ideal_planes:
                xx, yy = np.meshgrid(np.arange(-1, 1, .2), np.arange(-1, 1, .2))
                z = (-plane[0] * xx - plane[1] * yy - plane[3]) * 1. / plane[2]
                ax.plot_surface(xx, yy, z, alpha=.2)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()
