# Can We Use Mixture Models to Predict Market Bottoms?

#### This repo contains the code and supporting data to the webinar presented via QuantInsti.com April 25, 2017

Clone or download the repo
***
To import the GMM prediction dataframe and output

 model_data = [data/GMM_Results_TidyData.h5](data/GMM_Results_TidyData.h5)
```python
import pandas as pd

df = pd.read_hdf(model_data, 'table')
print(df.head())
```
![tidydf_head_github](https://cloud.githubusercontent.com/assets/7452471/25408806/81bc69b0-29cc-11e7-9686-a52d6efce113.png)

_lookback of `999` indicates an expanding lookback window._ 
***
To import the raw return data:

market_data = [data/mixture_model_merged_data_03.h5](data/mixture_model_merged_data_03.h5)
```python
import pandas as pd

df = pd.read_hdf(market_data, 'table')
print(df.head())
```
![mktdata_head_github](https://cloud.githubusercontent.com/assets/7452471/25409128/f1d882aa-29cd-11e7-99ec-d5b9f45c6ad0.png)

___
To run the model yourself and experiment with parameters:

```python
import ModelRunner
import ResultEval 
import ModelPlots

DIR = '/YOUR/PROJECT/DIR/'

# Model Params
# ------------
f1 = 'TEDRATE' # ted spread
f2 = 'T10Y2Y' # constant maturity ten yer - 2 year
f3 = 'T10Y3M' # constant maturity 10yr - 3m
factors = [f1, f2, f3]

a, b = (.2, .7) # found via coarse parameter search
alpha = 0.99
max_iter = 50

init = 'random' # or 'kmeans'
nSamples = 1_000
year = 2007 # cutoff
lookback = 1 # years

# k = 3 # n_components
# step_fwd = 3 # days
step_fwds = [1, 2, 3, 5, 7, 10, 21]
ks = [2,3,5,7,9,13,17,21]

chosen_syms = ['SPY', 'QQQ', 'TLT', 'GLD', 'EFA', 'EEM']
for k in ks:
    for step in step_fwds:
        for mkt in chosen_syms:
            p('-'*79)
            p('fitting:', mkt)
            p(f'params: k = {k} | step = {step} | lookback = {lookback}')
            p('...')
            ft_cols = [mkt + '_lret'] + factors

            MR_kwds = dict(ft_cols=ft_cols, k=k, init=init, max_iter=max_iter)
            MR = ModelRunner.ModelRunner(data, **MR_kwds)
            dct = MR.prediction_cycle(year, alpha, a, b,
                                        nSamples, lookback, mkt)

            res = ResultEval.ResultEval(dct, step_fwd=step)
            event_dict = res.get_event_states()
            event = list(event_dict.keys())[1] #[1] # TOO_LOW

            post_events = res.get_post_events(event_dict[event])
            end_vals = res.get_end_vals(post_events)
            smry = res.create_summary(end_vals)

            p()
            p('*'*25)
            p(mkt, event.upper())
            p(smry.T)  

            mp = ModelPlots.ModelPlots(mkt, post_events, event, DIR, year)
            agg_tmp_df = mp._agg_temp_event_returns()            
            mp.plot_pred_results(dct['pred'], dct['year'], dct['a'], dct['b'])
            mp.plot_equity_curve(agg_tmp_df, benchmark=data['SPY_lret'])
            mp.plot_distplot(end_vals, smry)
            break
        break
    break
```

![spy-too-low-summary-github-sample](https://cloud.githubusercontent.com/assets/7452471/25411603/ec0c78e2-29da-11e7-8678-0c825d7e7486.png)

![jsu -spy-predictionplot--2017-04-25 17 14](https://cloud.githubusercontent.com/assets/7452471/25411615/f83e7638-29da-11e7-9a27-1a3d663bef9c.png)

![jsu -spy-too_low-equitycurve-2017-04-25 17 14](https://cloud.githubusercontent.com/assets/7452471/25411619/fee1b0fe-29da-11e7-9893-2de93c72172d.png)

![jsu -spy-too_low-distplot--2017-04-25 17 14](https://cloud.githubusercontent.com/assets/7452471/25411625/06050f8e-29db-11e7-882a-5e14d7570fa7.png)
