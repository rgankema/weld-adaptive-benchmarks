import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import argparse

from matplotlib.backends.backend_pdf import PdfPages

# Plot the results and save to PDF
def plot(df, pp):
    df = df.groupby("type").mean().reset_index()
    df = df.sort_values(["comp_time"])

    df.plot(kind='bar', x='type', y='comp_time')

    pp.savefig()
    pp.close()

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Micro benchmark for adaptive joins"
    )
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Path to input file")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Path to output file")
    cmdline_args = parser.parse_args()
    opt_dict = vars(cmdline_args)
    in_path = opt_dict['input']
    out_path = opt_dict['output']

    pp = PdfPages(out_path)
    df = pd.read_csv(in_path)
    plot(df, pp)