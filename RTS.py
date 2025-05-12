import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from NSS_Interpolation import *
import random
from scipy.interpolate import CubicSpline, PchipInterpolator

def RTS_Simulation_Unshifted(file,RHP,number_of_sims,shift,methodzcb,tenor_points):
    '''
    Follows the Regulated technical standards procedure of simulating the
    yield curve. We do not include the final step here. This is shifting the
    rates such that the mean over all simulations match the current expectations.
    :param file: This file contains the historical record of yield curves for the last 5 years.
    :param RHP: The recommended holding period or simulation time of the procedure.
    :param number_of_sims: Number of simulations to be performed in the RTS.
    :param shift: The shift you apply to the complete dataset to ensure positive rates.
    :param methodzcb: Boolean denoting whether we are working on the zero-coupon bond prices or not.
    :param tenor_points: The tenor points of the observed yield curve.
    :return: The simulated rates.
    '''

    if methodzcb == False:
        df_percentage = pd.read_excel(file, engine='openpyxl')
        df_full = df_percentage / 100
        df = df_full[tenor_points]
    if methodzcb == True:
        df = file

    current_rates = df.iloc[-1].to_numpy()
    shifted_df = df + shift
    # Calculating the logarithmic return over the historical data
    x = shifted_df.to_numpy()
    if np.any(x <= 0):
        print("Invalid values found:", x[x <= 0])
    log_returns = np.log(shifted_df / shifted_df.shift(1))
    # Drop the NA's in the first row
    log_returns.dropna(inplace=True)
    # We shift all log_returns such that for each tenor point, the mean is 0
    log_returns_shifted = log_returns - log_returns.mean()
    # We now calculate the covariance matrix
    cov_matrix = log_returns_shifted.cov()
    # We calculate the eigenvalues and eigenvectors
    eigenvalues_all, eigenvectors_all = np.linalg.eig(cov_matrix)
    # Selecting the largest eigenvalues and corresponding eigenvectors
    # argsort gives the indexes in ascending order, but we want descending
    eigenvectors = eigenvectors_all.real
    eigenvalues = eigenvalues_all.real
    idx = eigenvalues.argsort()[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]
    top_3_eigenvalues = sorted_eigenvalues[:3]
    top_3_eigenvectors = sorted_eigenvectors[:, :3]
    # We project the returns onto the 3 principal eigenvectors
    projected_returns = log_returns_shifted @ top_3_eigenvectors
    # We calculate the matrix of returns used for simulation
    returns_for_simulation = projected_returns @ top_3_eigenvectors.T
    # We have 250 number of observation days in a year
    # We have RHP number of years as the recommended holding period
    n = RHP * 250
    m = df.shape[1]
    unshifted_simulations = np.zeros([number_of_sims,m])
    chosen_returns = np.zeros([number_of_sims,n,m])
    for j in range(number_of_sims):
        sample_returns_for_all_tenor = returns_for_simulation.sample(n=n,replace=True)
        sample_array = sample_returns_for_all_tenor.to_numpy()
        chosen_returns[j,:,:] = sample_array
        for i in range(m):
            sampled_returns = sample_array[:,i]
            sum_sampled_returns= sum(sampled_returns)
            unshifted_simulations[j,i] = (current_rates[i]*np.exp(sum_sampled_returns)) - shift
    #Depending whether we are working on the zero-coupon dataset or directly on the yield
    #we have a different way of solving
    if methodzcb == False:
        unshifted_simulations_percentage = unshifted_simulations * 100
    if methodzcb == True:
        unshifted_simulations_percentage = ( ((1 / unshifted_simulations)**(1 / tenor_points))  - 1 ) * 100
    return unshifted_simulations_percentage,chosen_returns

def Direct_Sim(file,RHP,number_of_sims,shift,methodzcb,tenor_points):
    '''
    This function is exactly similar as the previous one, but negates the PCA.
    :param file: This file contains the historical record of yield curves for the last 5 years.
    :param RHP: The recommended holding period or simulation time of the procedure.
    :param number_of_sims: Number of simulations to be performed in the RTS.
    :param shift: The shift you apply to the complete dataset to ensure positive rates.
    :param methodzcb: Boolean denoting whether we are working on the zero-coupon bond prices or not.
    :param tenor_points: The tenor points of the observed yield curve.
    :return: The simulated rates.
    '''
    #Exactly similar as previous function, now without the pca steps.
    if methodzcb == False:
        df_percentage = pd.read_excel(file, engine='openpyxl')
        df = df_percentage / 100
    if methodzcb == True:
        df = file

    current_rates = df.iloc[-1].to_numpy()
    shifted_df = df + shift
    # Calculating the logarithmic return over the historical data
    log_returns = np.log(shifted_df / shifted_df.shift(1))
    # Drop the NA's in the first row
    log_returns.dropna(inplace=True)
    # We shift all log_returns such that for each tenor point, the mean is 0
    returns_for_simulation = log_returns - log_returns.mean()
    n = RHP * 250
    m = df.shape[1]
    unshifted_simulations = np.zeros([number_of_sims,m])
    chosen_returns = np.zeros([number_of_sims,n,m])
    for j in range(number_of_sims):
        sample_returns_for_all_tenor = returns_for_simulation.sample(n=n,replace=True)
        sample_array = sample_returns_for_all_tenor.to_numpy()
        chosen_returns[j,:,:] = sample_array
        for i in range(m):
            sampled_returns = sample_array[:,i]
            sum_sampled_returns= sum(sampled_returns)
            unshifted_simulations[j,i] = (current_rates[i]*np.exp(sum_sampled_returns)) - shift
    if methodzcb == False:
        unshifted_simulations_percentage = unshifted_simulations * 100
    if methodzcb == True:
        unshifted_simulations_percentage = ( ((1 / unshifted_simulations)**(1 / tenor_points))  - 1 ) * 100
    return unshifted_simulations_percentage,chosen_returns


def Expectation_Rates(RHP,file,tenor_points,data):
    '''
    Calculates what the current expectation might be following the expectation hypothesis.
    :param RHP: Recommended holding period or simulation time
    :param file: This file contains the observed yield curve. From this file we obtain current expectations.
    :param tenor_points: The tenor points of the observed yield curve.
    :param data: String denoting whether we are dealing with the European dataset or American.
    :return: The expected yield rates for the tenor points at the end of simulation time.
    '''
    #We first construct an interpolation of the yield curve to have all yield rates for every month.
    df = pd.read_excel(file, engine='openpyxl')
    YTM = df["Yield to maturity"]
    Maturities = df["Maturities"]
    if data == "Europe":
        beta0 = 3  # initial guess
        beta1 = -2  # initial guess
        beta2 = 10 # initial guess
        beta3 = -8  # initial guess
        lambda0 = 5  # initial guess
        lambda1 = 3  # initial guess

        NSS_Maturities = np.linspace(0.25, 30, 358)  # Every month starting from 3 months till 30 years.

        OptiParam = NSSMinimize(beta0, beta1, beta2, beta3, lambda0, lambda1, Maturities, YTM)

        NSS_Yields = NelsonSiegelSvansson(NSS_Maturities, OptiParam[0], OptiParam[1], OptiParam[2], OptiParam[3],
                                      OptiParam[4], OptiParam[5])
        Yields = NSS_Yields / 100
        int_matur = NSS_Maturities
    if data == "USA":
        chip = PchipInterpolator(Maturities, YTM)
        chip_Maturities = np.linspace(1/12, 30, 360)
        chip_Yields = chip(chip_Maturities)
        Yields = chip_Yields / 100
        int_matur = chip_Maturities

    n = len(tenor_points)
    current_expectation_rates = np.zeros(n)
    index_RHP = np.abs(int_matur - RHP).argmin()
    #index_RHP = np.where(NSS_Maturities == RHP)[0][0]
    Yield_RHP = Yields[index_RHP]
    for i in range(n):
        if RHP + tenor_points[i] > 30:
            current_expectation_rates[i] = Yields[-1]
        else:
            index_RHP_T = np.abs(int_matur - (RHP + tenor_points[i])).argmin()
            #index_RHP_T = np.where(NSS_Maturities == RHP + tenor_points[i])[0][0]
            Yield_RHP_T = Yields[index_RHP_T]
            current_expectation_rates[i] = (((1 + Yield_RHP_T) ** (RHP + tenor_points[i]) / (1 + Yield_RHP) ** (RHP)) ** (1/tenor_points[i])) - 1
    expected_rates = current_expectation_rates * 100
    return expected_rates

def RTS_Curve_Simulation(file_historical,file_current,RHP,tenor_points,number_of_sims,shift,data,methodzcb,methodpca):
    '''
    Simulates the yield curve following the RTS.
    :param file_historical: This file contains the historical record of yield curves for the last 5 years.
    :param file_current: This file contains the observed yield curve. From this file we obtain current expectations.
    :param RHP: Recommended holding period or simulation time
    :param tenor_points: The tenor points of the observed yield curve.
    :param number_of_sims: Number of simulations
    :param shift: The shift you apply to the complete dataset to ensure positive rates.
    :param data: String denoting whether we are dealing with the European dataset or American.
    :param methodzcb: Boolean denoting whether we are working on the zero-coupon bond prices or not.
    :param methodpca: Boolean denoting whehter we apply a PCA or not.
    :return: The simulated yield rates at the end of simulation time.
    '''
    if methodpca == True:
        unshifted_rates, chosen_returns = RTS_Simulation_Unshifted(file_historical,RHP,number_of_sims,shift,methodzcb,tenor_points)
    if methodpca == False:
        unshifted_rates, chosen_returns = Direct_Sim(file_historical,RHP,number_of_sims,shift,methodzcb,tenor_points)
    expected_rates = Expectation_Rates(RHP,file_current,tenor_points,data)
    current_mean = np.mean(unshifted_rates,axis=0)
    shifted_rates = unshifted_rates - current_mean + expected_rates

    return shifted_rates,chosen_returns

#Extra analysis RTS:
#Direct spread (no logarithmic returns) --> Working on the zero-coupon bonds
#Shift analyis
#Tenor point analysis
#See impact on:  - Simulations
#                - Mrm classification

def Zero_Coupon_Bond_Conversion(historical_file,tenor_points):
    '''
    Converts the historical file of Yields to zero-coupon bond prices with 10000 notional (or price
    in basis points).
    :param historical_file: File that contains the historical record of the yield to maturity.
    :param tenor_points: The maturities of the yield to maturities of the historical file.
    :return: Dataframe with the zero coupon bond prices.
    '''
    df_percentage = pd.read_excel(historical_file, engine='openpyxl')
    df = df_percentage / 100
    yieldarray = df.to_numpy()
    #We apply the formula between yield and bond price
    #Making a distinction between sub 1y and over 1y.
    #oneyearindex = np.abs(tenor_points - 1).argmin()
    #zerocouponfile = np.zeros(df.shape)
    #zerocouponfile[:,:oneyearindex+1] = 1 / (1 + (yieldarray[:,:oneyearindex+1] * tenor_points[:oneyearindex+1]))
    #zerocouponfile[:,oneyearindex+1:] = 1 / ((1 + yieldarray[:,oneyearindex+1:]) ** tenor_points[oneyearindex+1:])
    zerocouponfile = 1 / ((1 + yieldarray) ** tenor_points)
    zerofile = pd.DataFrame(zerocouponfile)
    return zerofile













#Might leave this one out.
def RTS_Curve_Plot(file_historical,file_current,RHP,tenor_points,number_of_sims,number_of_curves,method,plotmean):
    shifted_rates = RTS_Curve_Simulation(file_historical,file_current,RHP,tenor_points,number_of_sims)
    random_selection = np.random.randint(0, number_of_sims, size=number_of_curves)
    if plotmean == False:
        for i in range(number_of_curves):
            Maturities = tenor_points
            YTM_percentage = shifted_rates[random_selection[i], :]
            if method == 'NSS':
                beta0 = 1  # initial guess
                beta1 = 2  # initial guess
                beta2 = -1  # initial guess
                beta3 = 5  # initial guess
                lambda0 = 1  # initial guess
                lambda1 = 10  # initial guess

                NSS_Maturities = np.linspace(0.25, 30, 358)  # Every month starting from 3 months till 30 years.

                OptiParam = NSSMinimize(beta0, beta1, beta2, beta3, lambda0, lambda1, Maturities, YTM_percentage)
                NSS_Yields = NelsonSiegelSvansson(NSS_Maturities, OptiParam[0], OptiParam[1], OptiParam[2], OptiParam[3],
                                          OptiParam[4], OptiParam[5])

                plt.plot(NSS_Maturities, NSS_Yields)
            else:
                plt.plot(Maturities,YTM_percentage)

    if plotmean == True:
        Maturities = tenor_points
        YTM_percentage = np.mean(shifted_rates,axis=0)

        beta0 = 1  # initial guess
        beta1 = 2  # initial guess
        beta2 = -1  # initial guess
        beta3 = 5  # initial guess
        lambda0 = 1  # initial guess
        lambda1 = 10  # initial guess

        NSS_Maturities = np.linspace(0.25, 30, 358)  # Every month starting from 3 months till 30 years.

        OptiParam = NSSMinimize(beta0, beta1, beta2, beta3, lambda0, lambda1, Maturities, YTM_percentage)
        NSS_Yields = NelsonSiegelSvansson(NSS_Maturities, OptiParam[0], OptiParam[1], OptiParam[2], OptiParam[3],
                                      OptiParam[4], OptiParam[5])

        plt.plot(NSS_Maturities, NSS_Yields)
    

    plt.xlabel("Residual maturity in years")
    plt.ylabel("Yield in %")
    plt.show()
    return None