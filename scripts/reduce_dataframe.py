import argparse
import os
import pandas as pd


def reduce_pkl(file_name: str, n: int) -> None:
    # Load DataFrame
    df = pd.read_pickle(file_name)

    if not isinstance(df, pd.DataFrame):
        raise TypeError("The pickle file does not contain a pandas DataFrame.")

    # Reduce to first N rows
    df_reduced = df.head(n)

    # Build output filename
    base, ext = os.path.splitext(file_name)
    output_file = f"{base}_reduced{ext}"

    # Save reduced DataFrame
    df_reduced.to_pickle(output_file)

    print(f"Saved {len(df_reduced)} rows to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Reduce a pandas DataFrame stored in a pickle file to the first N rows."
    )
    parser.add_argument(
        "--file_name",
        "-f",
        required=True,
        help="Path to the input .pkl file containing a pandas DataFrame",
    )
    parser.add_argument(
        "--n",
        "-n",
        type=int,
        required=True,
        help="Number of rows to keep",
    )

    args = parser.parse_args()

    if args.n <= 0:
        raise ValueError("N must be a positive integer.")

    reduce_pkl(args.file_name, args.n)


if __name__ == "__main__":
    main()
