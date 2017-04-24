import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
import affirm
from copy import deepcopy

sns.set(font_scale=1.25)
style_kwds = {'xtick.major.size': 3, 'ytick.major.size': 3,
              'font.family': u'CamingoCode', 'legend.frameon': True}
sns.set_style('white', style_kwds)


class ModelPlots():
    def __init__(self, mkt, post_events, event_state,
                 project_dir, file_fmt, year):
        """Class to visualize prediction results and summary

        Params:
            mkt : str(), symbol
            post_events : dict() of pd.Series()
            event_state : str(), 'too_high', 'too_low'
            project_dir : str()
            file_fmt : str(), file format
            year : int(), cutoff year
        """
        self.mkt = mkt
        self.post_events = post_events
        self.event_state = event_state
        self.DIR = project_dir + file_fmt + '/'
        self.file_fmt = file_fmt
        self.year = year
        self.today = pd.datetime.today().strftime("%Y-%m-%d %R")

    def _agg_temp_event_returns(self):
        agg_tmp = []
        d1 = deepcopy(self.post_events)
        for k in d1.keys():
            try:
                tmp = d1[k].copy()
                tmp.iloc[0] = 0  # set initial return to zero
                agg_tmp.append(tmp)
            except:
                continue
        agg_df = pd.concat(agg_tmp).cumsum()
        return agg_df

    def plot_equity_curve(self, return_data, benchmark=None):
        """Function to plot event timeline with equity curve second axis

        Params:
            benchmark : pd.Series() of benchmark returns
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        if isinstance(benchmark, pd.Series):
            benchmark = benchmark.loc[return_data.index[0]:]
            ax.plot(benchmark.index, np.exp(benchmark.cumsum()) - 1,
                    color='red', lw=1, marker='d', markersize=3,
                    label='SPY-B&H')
        ax.plot(return_data.index, np.exp(return_data.values) - 1,
                color='k', lw=1, marker='^', markersize=5,
                label=self.mkt + '-Algo')
        ax.axhline(y=0, color='k', ls='--', lw=3)
        ax.set_xlim(pd.to_datetime(str(self.year) + '-12-31'),
                    return_data.index[-1])
        ax.set_xlabel('Dates')
        ax.set_title(f"{self.mkt}\nJohnsonSU Distribution | {self.event_state.upper()}\n{self.file_fmt}",
                     fontsize=16)
        ax.legend(loc='upper right', fontsize=11,
                  frameon=True).get_frame().set_edgecolor('blue')
        sns.despine(offset=2)
        file_str = self.DIR + \
            f'[JSU]-{self.mkt}-{self.event_state.upper()}-{self.file_fmt}-EquityCurve-{self.today}.png'
        fig.savefig(file_str, dpi=300)
        return

    def plot_distplot(self, ending_values, summary):
        """Function to plot histogram of ending values"""
        colors = sns.color_palette('RdYlBu', 4)
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.distplot(pd.DataFrame(ending_values),
                     bins=15, color=colors[0], ax=ax,
                     kde_kws={"color": colors[3]},
                     hist_kws={"color": colors[3], "alpha": 0.35})
        ax.axvline(x=float(summary['mean'][0]), label='mean',
                   color='dodgerblue', lw=3, ls='-.')
        ax.axvline(x=float(summary['median'][0]), label='median',
                   color='red', lw=3, ls=':')
        ax.axvline(x=0, color='black', lw=1, ls='-')
        ax.legend(loc='upper right', fontsize=11,
                  frameon=True).get_frame().set_edgecolor('blue')
        sns.despine(offset=2)
        ax.set_title(
            f"{self.mkt}\nJohnsonSU Distribution | {self.event_state.upper()}\n{self.file_fmt}", fontsize=16)
        file_str = self.DIR + \
            f'[JSU]-{self.mkt}-{self.event_state.upper()}-{self.file_fmt}-Distplot--{self.today}.png'
        fig.savefig(file_str, dpi=300)
        return

    def plot_pred_results(self, df, year, a, b):
        """Fn: plot prediction results and confidence intervals"""
        # colorblind safe palette http://colorbrewer2.org/
        colors = sns.color_palette('RdYlBu', 4)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(df.index, df.tgt, alpha=0.85, s=20,
                   c=[colors[1] if x == 1 else colors[0] for x in df['in_rng']])
        df['high_ci'].plot(ax=ax, alpha=0.65, marker='.', color=colors[2])
        df['low_ci'].plot(ax=ax, alpha=0.65, marker='.', color=colors[3])
        ax.set_xlim(df.index[0], df.index[-1])

        nRight = df.query('in_rng==1').shape[0]
        accuracy = nRight / df.shape[0]
        ax.set_title('{:^10}\n[JSU] cutoff year: {} | accuracy: {:2.2%} | errors: {} | a={}, b={}\n{}'
                     .format(self.mkt, year, accuracy, df.shape[0] - nRight, a, b, self.file_fmt))

        win_kwds = dict(color="white", marker='o', markersize=5)
        in_ = mpl.lines.Line2D(range(1), range(1),
            markerfacecolor=colors[1], **win_kwds)
        out_ = mpl.lines.Line2D(range(1), range(1),
            markerfacecolor=colors[0], **win_kwds)
        ci_kwds = dict(color="white", marker='.', markersize=10)
        hi_ci = mpl.lines.Line2D(range(1), range(
            1), markerfacecolor=colors[2], **ci_kwds)
        lo_ci = mpl.lines.Line2D(range(1), range(
            1), markerfacecolor=colors[3], **ci_kwds)
        leg = ax.legend([in_, out_, hi_ci, lo_ci], ["in", "out", 'high_ci', 'low_ci'],
                        loc="center left", bbox_to_anchor=(1, 0.85), numpoints=1)
        leg.get_frame().set_edgecolor('blue')
        sns.despine(offset=2)
        file_str = self.DIR + \
            f'[JSU]-{self.mkt}-{self.file_fmt}-PredictionPlot--{self.today}.png'
        fig.savefig(file_str, dpi=300, bbox_inches="tight")
        return
