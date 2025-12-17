import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_xlsx", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # Excel loader (required by exam)
    df = pd.read_excel(args.input_xlsx, engine="openpyxl")

    df.to_csv(args.output_csv, index=False, encoding="utf-8")

    print("CSV written to:", args.output_csv)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

if __name__ == "__main__":
    main()
