import pandas as pd
import numpy as np
from scipy.stats import norm
from statsmodels.formula.api import ols
from tqdm.notebook import tqdm
from itertools import zip_longest



def calc_performance(subj_ID, data_all):

    """
    Calculate the baseline performance overall, useful for excluding participants with below chance performance
    :param subj_ID: the participant-unique ID
    :param data_all: the dataframe containing the individual participants' trial-level reponses
    :return: a participant's performance measures (hit, false alarm, dprime) for each duration
    """
    curr_sub_df = data_all[data_all['ID_REDACT'] == subj_ID]
    curr_sub_df['img_duration'] = [int(k[-1]) for k in curr_sub_df['Condition'].str.split('_')]
    duration_list = curr_sub_df['img_duration'].unique()

    curr_perf_df = pd.DataFrame(columns=['sub', 'duration', 'hit', 'fa', 'dprime'])

    curr_dur_df_yes = curr_sub_df[curr_sub_df['TrialType'] == 'YES']

    curr_dur_df_no = curr_sub_df[curr_sub_df['TrialType'] == 'NO']
    curr_perf_df.loc[len(curr_perf_df)] = {'sub': subj_ID,
                                           'duration': 'all',
                                           'hit': np.round(
                                               np.sum(curr_dur_df_yes['response'] == 'YES') / len(curr_dur_df_yes), 3),
                                           'fa': np.round(
                                               np.sum(curr_dur_df_no['response'] == 'YES') / len(curr_dur_df_no), 3),
                                           'dprime': np.round(norm.ppf(
                                               np.sum(curr_dur_df_yes['response'] == 'YES') / len(
                                                   curr_dur_df_yes)) - norm.ppf(
                                               np.sum(curr_dur_df_no['response'] == 'YES') / len(curr_dur_df_no)), 3)

                                           }

    for duration in duration_list:
        curr_dur_df_yes = curr_sub_df[(curr_sub_df['img_duration'] == duration) &
                                      (curr_sub_df['TrialType'] == 'YES')]

        curr_dur_df_no = curr_sub_df[(curr_sub_df['img_duration'] == duration) &
                                     (curr_sub_df['TrialType'] == 'NO')]

        curr_perf_df.loc[len(curr_perf_df)] = {'sub': subj_ID,
                                               'duration': duration,
                                               'hit': np.round(
                                                   np.sum(curr_dur_df_yes['response'] == 'YES') / len(curr_dur_df_yes),
                                                   3),
                                               'fa': np.round(
                                                   np.sum(curr_dur_df_no['response'] == 'YES') / len(curr_dur_df_no),
                                                   3),
                                               'dprime': np.round(norm.ppf(
                                                   np.sum(curr_dur_df_yes['response'] == 'YES') / len(
                                                       curr_dur_df_yes)) - norm.ppf(
                                                   np.sum(curr_dur_df_no['response'] == 'YES') / len(curr_dur_df_no)),
                                                                  3)

                                               }

    return curr_perf_df


def calc_performance_target(subj_ID, data_all):
    """
    Calculate the performance for each of the 12 conditions
    :param subj_ID: the participant-unique ID :param
    data_all: the dataframe containing the individual participants' trial-level reponses
    :return: a participant's performance measures (hit, false alarm, dprime) for each unique combination of
    distinctiveness, reconstruction error and duration
    """

    curr_sub_df = data_all[data_all['ID_REDACT'] == subj_ID]

    condition_list = curr_sub_df['Condition'].unique()

    curr_perf_df = pd.DataFrame(columns=['sub', 'condition', 'hit', 'fa', 'dprime'])

    for condition in condition_list:
        curr_dur_df_yes = curr_sub_df[(curr_sub_df['Condition'] == condition) &
                                      (curr_sub_df['TrialType'] == 'YES')]

        curr_dur_df_no = curr_sub_df[(curr_sub_df['Condition'] == condition) &
                                     (curr_sub_df['TrialType'] == 'NO')]
        total_trials = len(curr_dur_df_yes) + len(curr_dur_df_no)

        curr_hit = np.round(np.sum(curr_dur_df_yes['response'] == 'YES') / len(curr_dur_df_yes), 3)
        curr_fa = np.round(np.sum(curr_dur_df_no['response'] == 'YES') / len(curr_dur_df_no), 3)

        if curr_hit == 1:
            curr_hit = 1 - 1 / (2 * total_trials)
        elif curr_hit == 0:
            curr_hit = 1 / (2 * total_trials)

        if curr_fa == 1:
            curr_fa = 1 - 1 / (2 * total_trials)
        elif curr_fa == 0:
            curr_fa = 1 / (2 * total_trials)

        curr_perf_df.loc[len(curr_perf_df)] = {'sub': subj_ID,
                                               'condition': condition,
                                               'hit': curr_hit,
                                               'fa': curr_fa,
                                               'dprime': np.round(norm.ppf(curr_hit) - norm.ppf(curr_fa), 3)

                                               }

    curr_perf_df['duration'] = [int(k[-1]) for k in curr_perf_df['condition'].str.split('_')]
    curr_perf_df['property'] = curr_perf_df['condition'].str[:19]
    curr_perf_df['Dist'] = [k[1] for k in curr_perf_df['property'].str.split('_')]
    curr_perf_df['RE'] = [k[3] for k in curr_perf_df['property'].str.split('_')]

    return curr_perf_df

def calc_performance_group(df, sub_list):
    """
    Takes a list of participants and calculate the baseline performance for each of them separated by durations
    :param df: the dataframe containing the individual participants' trial-level reponses
    :param sub_list: the list of participants to be included
    :return: a dataframe with performance measures from all the included participants
    """
    counter=0
    df_filterd = df[df['ID_REDACT'].isin(sub_list)]
    for curr_sub in sub_list:
        curr_df = calc_performance(curr_sub, df_filterd)
        if counter == 0:
            counter += 1
            all_sub_df = curr_df
        else:
            all_sub_df = all_sub_df.append(curr_df).reset_index(drop=True)
    return all_sub_df


# Summarize performance separately for each condition for a list of subjects
def calc_performance_target_group(df, sub_list):
    """

    Takes a list of participants and calculate the baseline performance for each of them separated by conditions
    :param df: the dataframe containing the individual participants' trial-level reponses
    :param sub_list: the list of participants to be included
    :return: a dataframe with performance measures from all the included participants
    """
    counter=0
    df_filterd = df[df['ID_REDACT'].isin(sub_list)]
    for curr_sub in sub_list:
        curr_df = calc_performance_target(curr_sub, df_filterd)
        if counter == 0:
            counter += 1
            all_sub_df = curr_df
        else:
            all_sub_df = all_sub_df.append(curr_df).reset_index(drop=True)
    return all_sub_df

def fit_ols_get_beta(df):
    """
    Fit a linear model relating exposure time to hit rate
    :param df: a dataframe for a given participant, with two columns, hit rates and duration (exposure time)
    :return: the estimated beta
    """
    model = ols(formula='hit~duration', data=df)
    results = model.fit()
    beta = results.params['duration']
    return beta

def run_bootstrapping(df, sub_list, n_iter=1000):
    """
    Run bootstrapping to to get a distribution of estimated relationship between exposure time and hit rates
    (sample with replacement)
    :param df: a dataframe with all the participants' performance data
    :param sub_list: the list of unique participants
    :param n_iter: the number of bootstrapping iterations
    :return: a list of difference values between the beta estimated for images with large reconstruction error values
    and that for images with small reconstruction error values
    """
    rng = np.random.RandomState(1)
    all_diff = []
    for curr_iter in tqdm(range(n_iter)):
        sampled_subs = rng.choice(sub_list, len(sub_list))
        curr_iter_beta_diff = []
        for ind, curr_sampled_sub in enumerate(sampled_subs):
            curr_df = df[df['sub']==curr_sampled_sub]
            # fit a beta
            RE_large_df = curr_df[curr_df['RE'] == 'Large']

            RE_large_model = ols(formula='hit~duration', data=RE_large_df)
            RE_large_results = RE_large_model.fit()
            RE_large_beta = RE_large_results.params['duration']

            RE_small_df = curr_df[curr_df['RE'] == 'Small']
            RE_small_model = ols(formula='hit~duration', data=RE_small_df)
            RE_small_results = RE_small_model.fit()
            RE_small_beta = RE_small_results.params['duration']

            curr_iter_beta_diff.append(RE_large_beta - RE_small_beta)

        all_diff.append(np.mean(curr_iter_beta_diff))
    return all_diff

# For graphing
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def summarize_img_hit_by_duration(df, duration_list):
    """
    Summarize the hit rate for each image, separate for each duration
    :param df: a dataframe with all participants' responses to targets only
    :param duration_list: the list exposure times
    :return: a dataframe with the hit rate calculated for each image separately for each duration
    """

    df['img_duration'] = [int(k[-1]) for k in df['Condition'].str.split('_')]

    img_hit_by_duration = pd.DataFrame(columns=['Image index', 'hit', 'duration', 'num of responses'])
    for curr_ind, curr_duration in enumerate(duration_list):
        curr_df = df[df['img_duration'] == curr_duration]
        curr_img_hit = curr_df.groupby(by='test_img')['response_num'].mean().reset_index().rename(
            columns={'response_num': 'hit'})
        curr_img_resp_count = curr_df.groupby(by='test_img')['response_num'].count().reset_index().rename(
            columns={'response_num': 'num of responses'})

        curr_sum_df = curr_img_hit.merge(curr_img_resp_count)
        curr_sum_df['duration'] = curr_duration
        curr_sum_df['Image index'] = [int(k[0]) for k in curr_sum_df['test_img'].str.split('_')]

        if curr_ind == 0:
            img_hit_by_duration = curr_sum_df
        else:
            img_hit_by_duration = img_hit_by_duration.append(curr_sum_df).reset_index(drop=True)

    return img_hit_by_duration


def stack_data_for_bootstrapping(sub_list, df):
    """
    Restack the dataframe given a group of sampled participants
    :param sub_list: the list of sampled participants (i.e., could have duplicates because we sample with replacement)
    :param df: a dataframe with all participants' responses to targets only
    :return: a dataframe with responses from the sampled participants
    """
    for curr_ind, curr_sub in enumerate(sub_list):
        curr_sub_data = df[df['ID_REDACT'] == curr_sub]

        if curr_ind == 0:
            sampled_data = curr_sub_data
        else:
            sampled_data = sampled_data.append(curr_sub_data)

    return sampled_data

