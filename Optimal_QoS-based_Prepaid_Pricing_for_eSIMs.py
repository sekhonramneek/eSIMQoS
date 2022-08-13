
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline, BSpline

import numpy as np
import random


big_data = pd.read_csv('data_usage.csv')

big_data.BUNDLEID.unique()

big_data.groupby('BUNDLEID').mean().sort_values(by=['INITIALBALANCELIMIT'])

ids = big_data.BUNDLEID.unique()

# Insert Qmax for different data plans based on Simulated value.
qmax = [4,4,4,4,4,8,4,6,6,6,6,6,8,8,8,8,10,10,10,10,10]
map_dict = {k:v for k,v, in zip(ids,qmax)}

big_data['qmax'] = big_data['BUNDLEID'].map(map_dict)
big_data['qreq'] = big_data['qmax'].apply(lambda x: random.randint(1, x))
big_data.to_csv('qreq_data_usage.csv', index=False)

big_data

def obtain_x(data_usage_eval):
    '''
    Calculates X for expired and exhausted prepaid plans for all users.
    If the plan is expired, X is the fraction of data used before expiration.
    If the plan is exhausted, X is the fraction of time taken over the duration.

    :param: data_usage_eval: Accepts the dataframe with the relevant data and fields.
    :return: data_usage_eval: Dataframe with updated fields.
    '''

    # Insert Column X into the datafram
    # If headers do not exist, insert them
    if not 'X' in data_usage_eval.columns:
        data_usage_eval.insert(loc=6, column='X', value=' ')

    # Calculate fraction X
    for i in range(len(data_usage_eval)):
        if data_usage_eval.at[i, 'Bundle_Status'] == 'Bundle_Expired':
            data_usage_eval.at[i, 'X'] = data_usage_eval.at[i, 'Usage'] / data_usage_eval.at[i, 'INITIALBALANCELIMIT']
        elif data_usage_eval.at[i, 'Bundle_Status'] == 'Bundle_Exhausted':
            data_usage_eval.at[i, 'X'] = data_usage_eval.at[i, 'Time_Taken'] / data_usage_eval.at[i, 'Duration']

    data_usage_eval['X'] = pd.to_numeric(data_usage_eval['X'])

    return data_usage_eval

big_data = obtain_x(big_data)

def expired_x_graph_display(data_usage_eval, bundle_type='Bundle_Expired'):
    '''
    Displays the graph for the fraction of data used before expiration,
    against the different plans from the provider.

    :param data_usage_eval: Accepts the dataframe with the relevant data and fields.
    :output: Shows the graph
    '''

    # Get data for expired plans only
    if (bundle_type=='Bundle_Expired'):
        exp_prepaid = data_usage_eval.loc[data_usage_eval['Bundle_Status'] == 'Bundle_Expired']
    elif(bundle_type=='Bundle_Exhausted'):
        exp_prepaid = data_usage_eval.loc[data_usage_eval['Bundle_Status'] == 'Bundle_Exhausted']

    # Group by the Bundle ID and get the mean of each field
    group_prepaid_mean = exp_prepaid.groupby(['BUNDLEID']).mean().reset_index()

    sns.set_style('darkgrid')
    sns.set(rc={'figure.figsize':(11.7,8.27)})

    x = group_prepaid_mean.index
    y = group_prepaid_mean['X']

    ax = sns.barplot(x=x, y=y, data=group_prepaid_mean, palette="Blues_d")
    ax.set(xlabel='Different Prepaid Plans ', ylabel='Fraction of data used before expiration (x)')
    plt.show()

    return x, y

x_exhausted, y_exhausted=expired_x_graph_display(big_data, bundle_type='Bundle_Exhausted')

x_expired,y_expired =expired_x_graph_display(big_data, bundle_type='Bundle_Expired')

def get_profit(kp_list, data_usage_eval):
    '''
    Obtains the profit for traditional and personalized plans based
    on the profit margin

    :param: data_usage_eval: Accepts the dataframe with the relevant data and fields.
    :output: Saves the profits for exhausted and expired plans for both traditional and personalized,
    this is done to calculate the profit ratios later on.
    '''

    exp_personalized = dict()
    exp_traditional = dict()
    exh_personalized = dict()
    exh_traditional = dict()

    # If headers do not exist, insert them
    if not set(['Personalized_Profit','Traditional_Profit']).issubset(data_usage_eval.columns):
        data_usage_eval.insert(loc=7, column='Personalized_Profit', value=' ')
        data_usage_eval.insert(loc=8, column='Traditional_Profit', value=' ')

    # For each profit margin in the range of profit margins
    for kp in kp_list:
        print("Profit Margin: ", kp)
        margin = 1 / (1+kp)

        # Personalized profit and the traditional profit calculated for QoS-based Service Plans.
        for i in range(len(data_usage_eval)):
            if data_usage_eval.at[i, 'X'] <= (data_usage_eval['qmax'][i]/data_usage_eval['qreq'][i])*margin:
                # X * kp
                data_usage_eval.at[i, 'Personalized_Profit'] = (data_usage_eval.at[i, 'X'] * kp * data_usage_eval['qreq'][i])
                data_usage_eval.at[i, 'Traditional_Profit'] = 0
            elif data_usage_eval.at[i, 'X'] > margin:
                # 1 - X
                data_usage_eval.at[i, 'Traditional_Profit'] = data_usage_eval['qmax'][i]*(1 - data_usage_eval.at[i, 'X'])
                data_usage_eval.at[i, 'Personalized_Profit'] = 0

        data_usage_eval['Personalized_Profit'] = pd.to_numeric(data_usage_eval['Personalized_Profit'], errors='coerce')
        data_usage_eval['Traditional_Profit'] = pd.to_numeric(data_usage_eval['Traditional_Profit'], errors='coerce')

        # Separate expired and exhausted data
        expired = data_usage_eval.loc[data_usage_eval['Bundle_Status'] == 'Bundle_Expired']
        exhausted = data_usage_eval.loc[data_usage_eval['Bundle_Status'] == 'Bundle_Exhausted']

        # Sum the personalized and traditional profits calculated for expired plans
        exp_personalized[kp] = expired['Personalized_Profit'].sum()
        exp_traditional[kp] = expired['Traditional_Profit'].sum()

        # Sum the personalized and traditional profits calculated for expired plans
        exh_personalized[kp] = exhausted['Personalized_Profit'].sum()
        exh_traditional[kp] = exhausted['Traditional_Profit'].sum()

        print("Expired Personalized Profit: ", exp_personalized[kp])
        print("Expired Traditional Profit: ", exp_traditional[kp])
        print("Exhausted Personalized Profit: ", exh_personalized[kp])
        print("Exhausted Traditional Profit: ", exh_traditional[kp])
        print("\n")

    return exp_personalized, exp_traditional, exh_personalized, exh_traditional

kp_list = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]

# Save profits for personalized and traditional plans
exp_personalized, exp_traditional, exh_personalized, exh_traditional = get_profit(kp_list, big_data)

def get_profit_ratios(kp_list, exp_personalized, exp_traditional, exh_personalized, exh_traditional):
    '''
    Calculates the profit ratios using the personalized and
    traditonal profits previously calculated.

    :param: kp_list: Range of profit margins from 0.02 - 0.15
    :param: exp_personalized: Personalized profits for the range of profit margins [0.02 - 0.15] for expired fields.
    :param: exp_traditional: Traditional profits for the range of profit margins [0.02 - 0.15] for exhausted fields.
    :param: exh_personalized: Personalized profits for the range of profit margins [0.02 - 0.15] for expired fields.
    :param: exh_traditional: Traditional profits for the range of profit margins [0.02 - 0.15] for exhausted fields.

    :return: exp_ratios: Expired Profit Ratios.
    :return: exp_pers_profits: Personalized Profits for Expired Plans.
    :return: exp_trad_profits: Traditional Profits for Expired Plans.
    :return: exh_ratios: Exhausted Profit Ratios.
    :return: exh_pers_profits: Personalized Profits for Exhausted Plans.
    :return: exh_trad_profits: Traditional Profits for Exhausted Plans.
    '''
    exp_pers_profits = []
    exp_trad_profits = []
    exh_pers_profits = []
    exh_trad_profits = []
    exp_ratio = []
    exh_ratio = []

    # Get profits and profit ratios for expired and exhausted plans
    for x in kp_list:
        # Personalized profits for expired plans
        exp_pers_profits.append(exp_personalized[x])
        # Traditional profits for expired plans
        exp_trad_profits.append(exp_traditional[x])
        # Personalized profits for exhausted plans
        exh_pers_profits.append(exh_personalized[x])
        # Traditional profits for exhausted plans
        exh_trad_profits.append(exh_traditional[x])

        # Profit ratios for expired plans
        exp_ratio.append(exp_personalized[x]/exp_traditional[x])
        # Profit ratios for exhausted plans
        exh_ratio.append(exh_personalized[x]/exh_traditional[x])

    # Expired Data
    exp_ratios = np.array(exp_ratio)
    exp_pers_profits = np.array(exp_pers_profits)
    exp_trad_profits = np.array(exp_trad_profits)

    # Exhausted Data
    exh_ratios = np.array(exh_ratio)
    exh_pers_profits = np.array(exh_pers_profits)
    exh_trad_profits = np.array(exh_trad_profits)

    return exp_ratios, exp_pers_profits, exp_trad_profits, exh_ratios, exh_pers_profits, exh_trad_profits

exp_ratios, exp_pers_profits, exp_trad_profits, exh_ratios, exh_pers_profits, exh_trad_profits = get_profit_ratios(kp_list, exp_personalized, exp_traditional, exh_personalized, exh_traditional)

def profit_ratio_graph(kp_list, ratio, personalized, traditional):
    '''
    Calculates the profit ratios using the personalized and
    traditonal profits previously calculated.

    :param: kp_list: Range of profit margins from 0.02 - 0.15
    :param: ratio: Profit Ratios Previously Calculated
    :param: personalized: Personalized Profits
    :param: traditional: Traditional Profits

    :output: Profit Ratio Graph

    '''
    # Profit Margins
    kp = np.array(kp_list)

    # Range of profit margin from min to max
    xnew = np.linspace(kp.min(), kp.max(), 15)

    # Profit Ratios
    a_BSpline = make_interp_spline(kp, ratio)

    # Personalized Profits
    pers_BSpline = make_interp_spline(kp, personalized)

    # Traditional Profits
    trad_BSpline = make_interp_spline(kp, traditional)

    y_new = a_BSpline(xnew)
    y_pers = pers_BSpline(xnew)
    y_trad = trad_BSpline(xnew)

    # Set the x axis label
    plt.xlabel('Profit Margin (kp)')
    # Set the y axis label
    plt.ylabel('Profit Ratio (G)')

    # Plot the graph
    plt.plot(xnew, y_new)
    plt.show()

    return xnew, y_new

exp_ratios

xnew_exp, y_new_exp  = profit_ratio_graph(kp_list, exp_ratios, exp_pers_profits, exp_trad_profits)

xnew_exh, y_new_exh = profit_ratio_graph(kp_list, exh_ratios, exh_pers_profits, exh_trad_profits)
