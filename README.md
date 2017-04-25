# Can We Use Mixture Models to Predict Market Bottoms?

#### This repo contains the code and supporting data to the webinar presented via QuantInsti.com April 25, 2017

Clone or download the repo

To import the GMM prediction dataframe and output

 model_data = [data/GMM_Results_TidyData.h5](data/GMM_Results_TidyData.h5)
```python
import pandas as pd

df = pd.read_hdf(model_data, 'table')
print(df.head())
```
![tidydf_head_github](https://cloud.githubusercontent.com/assets/7452471/25408806/81bc69b0-29cc-11e7-9686-a52d6efce113.png)

lookback of `999` indicates an expanding lookback window. 

To import the raw return data:

market_data = [data/mixture_model_merged_data_03.h5](data/mixture_model_merged_data_03.h5)
```python
import pandas as pd

df = pd.read_hdf(market_data, 'table')
print(df.head())
```
![mktdata_head_github](https://cloud.githubusercontent.com/assets/7452471/25409128/f1d882aa-29cd-11e7-99ec-d5b9f45c6ad0.png)

