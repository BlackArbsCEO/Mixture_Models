import pandas as pd
import numpy as np

from copy import deepcopy
p = print

class ResultEval():
    def __init__(self, data, step_fwd):
        """Class to evaluate prediction results

        Params:
            data : dict() containing results of ModelRunner()
            step_fwd : int(), number of days to evalute post event
        """
        self.df = data['pred'].copy().reset_index()
        self.step_fwd = step_fwd

    def get_event_states(self):
        """Function to get event indexes
        Index objects must be called 'too_high', 'too_low'

        Returns:
            dict() : values are index objects
        """
        too_high = self.df.query("tgt > high_ci").index
        too_low = self.df.query("tgt < low_ci").index
        return {'too_high': too_high, 'too_low': too_low}

    def get_post_events(self, event):
        """Function to return dictionary where key, value is integer
        index, and Pandas series consisting of returns post event

        Params:
            df : pd.DataFrame(), prediction df
            event : {array-like}, index of target returns that exceed CI high or low
            step_fwd : int(), how many days to include after event
        Returns:
            after_event : dict() w/ values = pd.Series()
        """
        after_event = {}
        for i in range(len(event)):
            tmp_ret = self.df.ix[event[i]:event[
                i] + self.step_fwd, ['Dates', 'tgt']]
            # series of returns with date index
            after_event[i] = tmp_ret.set_index('Dates', drop=True).squeeze()
        return after_event

    def agg_temp_event_returns(self, post_events):    
        agg_tmp = []
        d1 = deepcopy(post_events)
        for k in d1.keys():
            try:
                tmp = d1[k].copy()
                tmp.iloc[0] = 0 # set initial return to zero 
                agg_tmp.append(tmp)
            except: continue
        agg_df = pd.concat(agg_tmp).cumsum()
        return agg_df

    def get_end_vals(self, post_events):
        """Function to sum and agg each post events' returns"""
        pe_copy = copy.deepcopy(post_events)
        end_vals = []
        for k in pe_copy.keys():
            try:
                tmp = pe_copy[k].copy()
                tmp.iloc[0] = 0  # set initial return to zero
                end_vals.append(tmp.sum())
            except Exception as e:
                p()
                p(e)
                #p(self.mkt + ' Error:', e)
                #p(' at k index: ', k)
        return end_vals

    def create_summary(self, end_vals):
        """Function to take ending values and calculate summary
        Will fail if count of ending values (>0) or (<0) is less than 1
        """
        gt0 = [x for x in end_vals if x > 0]
        lt0 = [x for x in end_vals if x < 0]
        assert len(gt0) > 1
        assert len(lt0) > 1
        summary = (pd.DataFrame(index=['value'])
                   .assign(mean=f'{np.mean(end_vals):.4f}')
                   .assign(median=f'{np.median(end_vals):.4f}')
                   .assign(max_=f'{np.max(end_vals):.4f}')
                   .assign(min_=f'{np.min(end_vals):.4f}')
                   .assign(gt0_cnt=f'{len(gt0):d}')
                   .assign(lt0_cnt=f'{len(lt0):d}')
                   .assign(sum_gt0=f'{sum(gt0):.4f}')
                   .assign(sum_lt0=f'{sum(lt0):.4f}')
                   .assign(sum_ratio=f'{sum(gt0) / abs(sum(lt0)):.4f}')
                   .assign(gt_pct=f'{len(gt0) / (len(gt0) + len(lt0)):.4f}')
                   .assign(lt_pct=f'{len(lt0) / (len(gt0) + len(lt0)):.4f}')
                   )
        return summary
