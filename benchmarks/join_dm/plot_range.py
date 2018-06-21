import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import argparse

from matplotlib.backends.backend_pdf import PdfPages

# Colors for representing types
type_colors = {
    'Normal': 'blue',
    'Bloom Filter': 'green',
    'Adaptive': 'orange',
    'Lazy': 'red'
}

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
    sf_vals = df.sf.unique()

    w = len(thread_vals)
    h = len(sf_vals)
    fig, axs = plt.subplots(h, w, figsize=(16, 24))

    y = 0
    for sf in sf_vals:
        # Ensure that each row shares the same y-axis
        #max_time = df.loc[df.sf == sf].norm_total_time.max()
        #for x in range(w):
        #    axs[y,x].set_ylim([0, max_time * 1.05])
        x = 0

        for th in thread_vals:
            plots = []
            for ty in type_vals:
                g = df.loc[df.type == ty].loc[df.threads == th].loc[df.sf == sf].groupby('s_hit').median().reset_index()
                z = df.loc[df.type == ty].loc[df.threads == th].loc[df.sf == sf]

                plot = axs[y,x].errorbar(g.s_hit, g.norm_total_time, c=type_colors[ty])
                axs[y,x].scatter(z.s_hit, z.norm_total_time, s=3, c=type_colors[ty])
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