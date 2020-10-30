import sys

if not sys.path.__contains__("D:\\Projects\\Python\\plane_clustering"):
    sys.path.append("D:\\Projects\\Python\\plane_clustering")
from src.DatasetGenerator import DatasetGenerator
from src.Visualizer import Visualizer
import pandas as pd

if __name__ == '__main__':
    generator = DatasetGenerator(3, 3, 200, 0.2, 5184)
    # dataset, description = generator.generate_dataset()
    # generator.generate_and_save_dataset("../datasets/test_set/")
    dataset, desc = generator.load_dataset("../datasets/test_set/")
    vis = Visualizer()
    for i in range(3):
        vis.draw_points(dataset[i]["points"], dataset[i]["labels"], dataset[i]["ideal_planes"])

    print("IDE Debug Breakpoint")
