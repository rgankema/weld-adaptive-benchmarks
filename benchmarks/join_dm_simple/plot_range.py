import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import argparse

from matplotlib.backends.backend_pdf import PdfPages

# Get a dataframe from CSV and preprocess it
def get_dataframe(path):
    df = pd.read_csv(path)

    def normalize_total_time(group):
        group['norm_total_time'] = group.exec_time + group.comp_time.median()
        return group

    df['total_time'] = df.comp_time + df.exec_time
    df = df.groupby('type').apply(normalize_total_time).reset_index()

    return df

# Plot the results and save to PDF
def plot(df, pp):
    type_vals = df.type.unique()
    thread_vals = df.threads.unique()

    w = len(thread_vals)
    h = 1
    fig, axs = plt.subplots(h, w, figsize=(16, 8))

   
    x = 0
    for th in thread_vals:
        # Ensure that each row shares the same y-axis
        max_time = df.exec_time.max()

        plots = []
        for ty in type_vals:
            g = df.loc[df.type == ty].loc[df.threads == th].groupby('s_hit').median().reset_index()
            z = df.loc[df.type == ty].loc[df.threads == th]

            plot = axs[x].errorbar(g.s_hit, g.exec_time)
            axs[x].scatter(z.s_hit, z.exec_time, s=3)
            axs[x].set_ylim([0, max_time * 1.05])
            plots.append(plot)

        axs[x].set_title('threads=%d' % th)
        axs[x].legend([p[0] for p in plots], type_vals)
        x += 1

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
    df = get_dataframe(in_path)
    plot(df, pp)