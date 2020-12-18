import sys

if not sys.path.__contains__("D:\\Projects\\Python\\plane_clustering"):
    sys.path.append("D:\\Projects\\Python\\plane_clustering")
from src.Clustering import variant_01, variant_03
from src.DatasetGenerator import DatasetGenerator
from src.Visualizer import Visualizer
import pandas as pd

if __name__ == '__main__':
    generator = DatasetGenerator(100, 3, 200, 0.1, 2187)
    # dataset, desc = generator.generate_dataset()
    generator.generate_and_save_dataset("../datasets/dataset01/")
    dataset, desc = generator.load_dataset("../datasets/test_set/")

    for instance in dataset:
        # result01 = variant_01(instance, desc, 3)
        result03 = variant_03(instance, desc)

    print("IDE Debug Breakpoint")
