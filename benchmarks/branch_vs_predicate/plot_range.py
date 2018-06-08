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
def plot(df, pp, outliers=False):
    type_vals = df.type.unique()
    thread_vals = df.threads.unique()
    sf_vals = df.sf.unique()

    w = len(thread_vals)
    h = len(sf_vals)
    fig, axs = plt.subplots(h, w, figsize=(16, 8 * h))

    y = 0
    for sf in sf_vals:
        # Set y-lim
        if outliers:
            max_time = df.loc[df.sf == sf].norm_total_time.max()
        else:
            max_time = df.loc[df.sf == sf].groupby(['type', 'threads', 'selectivity']).median().reset_index().norm_total_time.max()
        for x in range(w):
            axs[y,x].set_ylim([0, max_time * 1.05])
        x = 0

        for th in thread_vals:
            plots = []
            for ty in type_vals:
                g = df.loc[df.type == ty].loc[df.threads == th].loc[df.sf == sf].groupby('selectivity').median().reset_index()
                z = df.loc[df.type == ty].loc[df.threads == th].loc[df.sf == sf]

                plot = axs[y,x].errorbar(g.selectivity, g.norm_total_time)
                axs[y,x].scatter(z.selectivity, z.norm_total_time, s=3)
                plots.append(plot)

            axs[y,x].set_title('sf=%d, threads=%d' % (sf, th))
            axs[y,x].legend([p[0] for p in plots], type_vals)
            
            x += 1
            
        y += 1

    pp.savefig()
    pp.close()

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Micro benchmark for adaptive branchings vs predication"
    )
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Path to input file")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Path to output file")
    parser.add_argument('-s', '--show-outliers', required=False, action='store_true',
                        help='Plot all outliers')
    cmdline_args = parser.parse_args()
    opt_dict = vars(cmdline_args)
    in_path = opt_dict['input']
    out_path = opt_dict['output']
    show_outliers = opt_dict['show_outliers']

    pp = PdfPages(out_path)
    df = get_dataframe(in_path)
    plot(df, pp, show_outliers)