import argparse
import os
import json

from distributed import Client, LocalCluster
import dask.dataframe as dd

from submission.preprocessors import create_preprocessor


def main(csv_dir, types_path, preproc_path, output_path):
    print("** Start preprocessing **")

    if not os.path.exists(output_path):
        # Read type & transform configs
        with open(types_path) as jf:
            dtypes = json.load(jf)

        with open(preproc_path) as jf:
            preproc = json.load(jf)

        with LocalCluster() as cluster:
            with Client(cluster) as client:
                df = dd.read_csv(os.path.join(csv_dir, '*.csv'), dtype=dtypes)

                # Separate categoricals
                cat_cols = [col for col, tp in dtypes.items() if tp == 'category' and col in df.columns]
                num_cols = [col for col in df.columns if col not in cat_cols]

                for desc in preproc["preprocessors"]:
                    p = create_preprocessor(desc)
                    df = p.apply(df)

                # {"column": "TailNum", "name": "FillValue", "value": "UNKNOW"},

                # Convert to known categoricals
                df = dd.get_dummies(df.categorize())
                print("Columns after preprocessing: ", df.columns)

                df.to_hdf(output_path, '/data')
    else:
        print("- file {} already exists, skipping preprocessing".format(output_path))

    print("** Finished preprocessing **")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess raw dataset and store the result in parquet format.')
    parser.add_argument('-i', '--input_dir', action="store", dest='csv_dir', type=str, required=True)
    parser.add_argument('-t', '--types_file', action="store", dest='types_path', type=str, required=True)
    parser.add_argument('-p', '--preproc_file', action="store", dest='preproc_path', type=str, required=True)
    parser.add_argument('-o', '--output_file', action="store", dest='output_path', type=str, required=True)

    args = parser.parse_args()

    main(args.csv_dir, args.types_path, args.preproc_path, args.output_path)
