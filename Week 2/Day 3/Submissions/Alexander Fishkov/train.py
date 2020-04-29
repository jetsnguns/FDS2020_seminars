import argparse
import json
import os
import time

from distributed import Client, LocalCluster

from submission.runner import get_experiment_runner

seed = 31337


def main(input_path, target_col, desc_path, output_path):
    print("** Start Training **")

    with open(desc_path) as fp:
        desc = json.load(fp)

    with LocalCluster(n_workers=4, threads_per_worker=5, host="192.168.1.32") as cluster:
        print(cluster)
        with Client(cluster) as client:
            print(client)
            r = get_experiment_runner(input_path, target_col, desc)
            r.run()

    print("** Finished Training **")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read preprocessed data in parquet format and train the models.')
    parser.add_argument('-i', '--input_file', action="store", dest='input_path', type=str, required=True)
    parser.add_argument('-t', '--target_col', action="store", dest='target_col', type=str, required=True)
    parser.add_argument('-d', '--desc_file', action="store", dest='desc_path', type=str, required=True)
    parser.add_argument('-o', '--output_file', action="store", dest='output_path', type=str, required=True)

    args = parser.parse_args()

    main(args.input_path, args.target_col, args.desc_path, args.output_path)
