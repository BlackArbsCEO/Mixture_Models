import pandas as pd
from pandas.tseries.offsets import Week
import numpy as np
import sklearn.mixture as mix


class ModelRunner():
    def __init__(self, data, **kwds):
        """Class to run mixture model model

        Params:
            data : pd.DataFrame()
            ft_cols : list() of feature columns str()
            k : int(), n_components
            max_iter : int(), max iterations
            init : str() {random, kmeans}
        """
        self.data = data
        self.ft_cols = kwds['ft_cols']
        self.k = kwds['k']
        self.max_iter = kwds['max_iter']
        self.init = kwds['init']

    def _run_model(self, data, bgm=None, **kwargs):
        """Function to run mixture model

        Params:
            data : pd.DataFrame() 
            ft_cols : list of str() 
            k : int(), n_components
            max_iter : int()
            init : str(), (random, kmeans)

        Returns:
            model : sklearn model object
            hidden_states : array-like, hidden states
        """
        X = data[self.ft_cols].values

        if bgm:
            model = mix.BayesianGaussianMixture(n_components=self.k,
                                                max_iter=self.max_iter,
                                                init_params=self.init,
                                                random_state=0,
                                                **kwargs,
                                                ).fit(X)
        else:
            model = mix.GaussianMixture(n_components=self.k,
                                        max_iter=self.max_iter,
                                        init_params=self.init,
                                        random_state=0,
                                        **kwargs,
                                        ).fit(X)

        hidden_states = model.predict(X)
        return model, hidden_states

    def _get_state_est(self, model, hidden_states):
        """Function to return estimated state mean and state variance

        Params:
            model : sklearn model object
            hidden_states : {array-like}
        Returns:
            mr_i : mean return of last estimated state
            mvar_i : model variance of last estimated state
        """
        # get last state
        last_state = hidden_states[-1]
        # first value is mean return for ith state
        # ft_cols in order ['ETF_lret', 'TEDRATE', 'T10Y2Y', 'T10Y3M']
        mr_i = model.means_[last_state][0]
        mvar_i = np.diag(model.covariances_[last_state])[0]
        return mr_i, mvar_i

    def _get_ci(self, mr_i, mvar_i, alpha, a, b, nSamples):
        """Function to sample confidence intervals 
            from the JohnsonSU distribution

        Params:
            mr_i : float()
            mvar_i : float()
            alpha : float()
            a : float()
            b : float() 
            nsamples : int()
            bins : int() 
        Returns:
            ci : tuple(float(), float()), (low_ci, high_ci) 
        """
        np.random.seed(123457)
        rvs_ = scs.johnsonsu.rvs(a, b, loc=mr_i, scale=mvar_i, size=nSamples)
        ci = scs.johnsonsu.interval(alpha=alpha, a=a, b=b,
                                    loc=np.mean(rvs_), scale=np.std(rvs_))

        #rvs_ = scs.norm.rvs(loc=mr_i, scale=mvar_i, size=nSamples)
        # ci = scs.norm.interval(alpha=alpha,
        #                       loc=np.mean(rvs_), scale=np.std(rvs_))
        return ci

    def prediction_cycle(self, *args, **kwargs):
        """Function to make walk forward predictions from cutoff year onwards

        Params:
            year : int(), cutoff year
            alpha : float()
            a : float()
            b : float() 
            nsamples : int()
        Returns:
            dict() :
                pred : pd.DataFrame()
                year : str()
                a, b : float(), float()
        """
        cutoff = year
        # index code takes advantage of pandas .ix datetime convention
        # when selecting by year only, pandas defaults to first day of year
        # when selecting an end point by year only, pandas defaults to
        #  last day of year
        train_df = self.data.ix[str(cutoff - lookback):str(cutoff)].dropna()
        # train_df stops one step before start of out of sample (oos) dataframe
        oos = self.data.ix[str(cutoff + 1):].dropna()
        # confirm that end index train_df is different than oos start index
        assert train_df.index[-1] != oos.index[0]

        # create pred list to hold tuple rows
        pred_list = []
        # 't' iterates through the oos index
        # this ensures our predictions are made using only info
        # available to us one step prior to actual return
        for t in tqdm(oos.index):
            # insample begins one step before oos df
            # at start insample equals train_df
            if t == oos.index[0]:
                insample = train_df

            # run model func to return model object and hidden states using
            # params
            model, hstates = self._run_model(insample, **kwargs)
            # get hidden state mean and variance
            mr_i, mvar_i = self._get_state_est(model, hstates)
            # get confidence intervals from sampled distribution
            low_ci, high_ci = self._get_ci(mr_i, mvar_i, alpha, a, b, nSamples)
            # append tuple row to pred list
            pred_list.append((t, hstates[-1], mr_i, mvar_i, low_ci, high_ci))
            # increment insample dataframe
            # insample increment includes the day 't'
            #   should it include up to t! is it subtle lookahead!
            #   not sure! i think it may be fine as is
            # trim dataframe
            insample = self.data.ix[t - lookback * 52 * Week():t]
            # note the cycle increments after we set the new insample

        cols = ['Dates', 'ith_state', 'ith_ret',
                'ith_var', 'low_ci', 'high_ci']
        pred_df = (pd.DataFrame(pred_list, columns=cols)
                   .set_index('Dates').assign(tgt=oos[mkt + '_lret']))

        # logic to see if error exceeds neg or pos CI
        pred_copy = pred_df.copy().reset_index()
        # Identify indices where target return falls between CI
        win = pred_copy.query("low_ci < tgt < high_ci").index
        # create list of binary variables representing in/out CI
        in_rng_list = [1 if i in win else 0 for i in pred_copy.index]
        # assign binary variables sequence to new column
        pred_df['in_rng'] = in_rng_list
        return {'pred': pred_df, 'year': year, 'a': a, 'b': b}
