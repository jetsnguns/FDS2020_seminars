import argparse
import json
import os
import time

from distributed import Client, LocalCluster

from submission.runner import get_experiment_runner

seed = 31337


def train(input_path, target_col, desc_path, output_path):
    """
    Perform CV and train the final model according to the description file.
    :param input_path: path to HDF file with processed dataset.
    :param target_col: name of the target column.
    :param desc_path: JSON model description file.
    :param output_path: serialize the final model to file. Currently unused.
    :return:
    """
    print("** Start Training **")

    with open(desc_path) as fp:
        desc = json.load(fp)

    with LocalCluster(n_workers=4, threads_per_worker=5) as cluster:
        print(cluster)
        with Client(cluster) as client:
            print(client)
            r = get_experiment_runner(input_path, target_col, desc)
            r.run()

    print("** Finished Training **")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read preprocessed data in HDF format and train the model')
    parser.add_argument('-i', '--input_file', action="store", dest='input_path', type=str, required=True,
                        help="Path to preprocessed data in HDF format")
    parser.add_argument('-t', '--target_col', action="store", dest='target_col', type=str, required=True,
                        help="Target column name")
    parser.add_argument('-d', '--desc_file', action="store", dest='desc_path', type=str, required=True,
                        help="Model description JSON file")
    parser.add_argument('-o', '--output_file', action="store", dest='output_path', type=str, required=True,
                        help="For future use: serialize the final model")

    args = parser.parse_args()

    train(args.input_path, args.target_col, args.desc_path, args.output_path)
