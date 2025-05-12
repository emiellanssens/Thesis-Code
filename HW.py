import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from NSS_Interpolation import *
from scipy.stats import norm
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline, PchipInterpolator

# We start by implementing the Black formula for pricing the swapions
# We have a dataframe with atm swaptions
# For implementing the formula we need:
# Strike of swaption
# Spot swap rate (is the same as the strike as we are dealing with ATM)
# Volatilities
# Expiry/Maturity
# Discount Factors

# EUROPE: STARTING SHORT RATE ON 29/10/2010 IS 0.00724
# USA:    STARTING SHORT RATE ON 13/11/2024 IS 0.0459

def BlackFormula(K,S0,avg_vol):
    '''
    Function that calculates the price of European payer swaptions using Black's formula.
    :param K: The strike of the swaption.
    :param S0: The starting swap rate (note that if we are dealing with ATM this will equal the strike)
    :param avg_vol: The implied volatility
    :return: The price of the European payer swaption
    '''
    #We create d1 and d2 and use the standard Gaussian cdf.
    d1 = (np.log(S0/K) + (avg_vol**2 / 2)) / avg_vol
    d2 = (np.log(S0/K) - (avg_vol**2 / 2)) / avg_vol
    price = (S0 * norm.cdf(d1)) - (K * norm.cdf(d2))
    return price

# We have to make some assumptions on the frequency of payments
# We can assume that a payment occurs every half-year (semi-annual)
# We determine our discount factors using the yield curve observed on 29/10/2010
# This works specifically for 29/10/2010
# For the USA we have

def DiscountFactors(yield_curve_file,data):
    '''
    We create the discountfactors or (zero-coupon bond prices) observed today, by interpolating the observed yield curve.
    :param yield_curve_file: File consist of values for zero-coupon bond prices on 29/10/2010
    :param data: String denoting whether we are dealing with the European dataset or American.
    :return: discountfactors for every month starting from 3 months of today until 30 years.
    '''
    #Read in the file
    df = pd.read_excel(yield_curve_file, engine='openpyxl')
    YTM = df["Yield to maturity"]
    Maturities = df["Maturities"]

    if data == "Europe":
        #The choice of initial guess for the NSS interpolation is necessary to find
        #optimal parameters.
        beta0 = 3  # initial guess
        beta1 = -2  # initial guess
        beta2 = 11  # initial guess
        beta3 = -8  # initial guess
        lambda0 = 5  # initial guess
        lambda1 = 3  # initial guess
        NSS_Maturities = np.linspace(0.25, 30, 358)  # Every month starting from 3 months till 30 years.
        OptiParam = NSSMinimize(beta0, beta1, beta2, beta3, lambda0, lambda1, Maturities, YTM)
        NSS_Yields = NelsonSiegelSvansson(NSS_Maturities, OptiParam[0], OptiParam[1], OptiParam[2], OptiParam[3],
                                      OptiParam[4], OptiParam[5])
        # The yields are denoted in percentage, so we transform them to decimals
        yields = NSS_Yields / 100
        act_mat = NSS_Maturities

    if data == "USA":
        chip = PchipInterpolator(Maturities, YTM)
        chip_Maturities = np.linspace(1 / 12, 30, 360)
        chip_Yields = chip(chip_Maturities)
        yields = chip_Yields / 100
        act_mat = chip_Maturities
    #We create the discountfactors based on the yields interpolated using NSS model
    discountfactors = np.zeros((len(act_mat)+1,2))
    #We manually add the discountfactor of today: 1
    discountfactors[0,:] = np.array([0,1])
    discountfactors[1:,0] = act_mat
    #The yields are denoted in percentage, so we transform them to decimals
    #We use the formula for annual interest rate --> zero-coupon bond price (discountfactor)
    dffac = 1 / ((1 + yields) ** act_mat)
    discountfactors[1:, 1] = dffac
    return discountfactors, yields

def InstaForward(disc_factors,r0):
    '''
    We create the instantaneous forward curve by the formula: f(t,T) = - d log(P(t,T))/dT
    using the discount factors (zero coupon bond prices) from before.
    :param disc_factors: The discount factors calculated using the previous function.
    :param r0: The instantaneous spot interest rate observed today.
    :return: f(t,T) of today, so f(0,T) for different maturities.
    '''
    x = disc_factors[:, 0]
    y = np.log(disc_factors[:, 1])
    dy_dx = np.gradient(y, x)
    f = -dy_dx
    #There is a small change on zero, so we fix it to be equal to the insta short rate.
    f[0] = r0
    return f

def SwaptionPriceMarket(swaption_file,yield_curve_file,data):
    '''
    This function calculates the price of the ATM European payer swaptions based on their
    implied Black volatilities. We use discount factors obtained from the previous functions.
    :param swaption_file: File containing the expiry, tenor, strike and implied volatility of European payer swaptions quoted on 29/10/2010.
    :param yield_curve_file: File containing the observed yield curve of 29/10/2010.
    :param data: String denoting whether we are dealing with the European dataset or American.
    :return: The prices of the European payer swaptions.
    '''
    #We start by loading in the discount factors
    discountfactors, _ = DiscountFactors(yield_curve_file,data)
    #We alsor read in the values for the swaptions.
    df = pd.read_excel(swaption_file, engine='openpyxl')
    expiries = df["Expiry"]
    maturities = df["Maturity"]
    #Volatilities and strikes are both quoted in percentage, thus change them to decimal notation
    volatilities = df["Vol"] / 100
    strikes = df["Strike"] / 100
    N = len(volatilities)
    swaption_prices = np.zeros(N)
    for i in range(N):
        #For every swaption, we compute the price using Black's formula
        T_alpha = expiries[i]
        Tenor = [T_alpha + j * 0.5 for j in range(2*maturities[i] + 1)]
        payment_times = Tenor[1:]
        time_vec = discountfactors[:,0]
        indices = np.argmin(np.abs(time_vec[:,None] - payment_times),axis=0)
        #indices = np.where(np.isin(discountfactors[:,0], payment_times))[0]

        df_fac = discountfactors[:,1]
        payment_df_fac = df_fac[indices]
        annuity = 0.5 * np.sum(payment_df_fac)
        avg_vol = volatilities[i] * np.sqrt(T_alpha)
        swaption_prices[i] = annuity * BlackFormula(strikes[i],strikes[i],avg_vol)
    return swaption_prices

# Now we want to implement the swaption pricer under Hull-White model
# Following Brigo-Mercurio we have: p.77
# t = 0, valuing until today.
# T = T_alpha (expiry of the swaption, start of tenor)
# t1, ..., t_n = T_beta (payment_times of swap)
# r* --> A and B functions and strikes and payment times
# Xi --> r*, A and B and payment times
# ZBP --> strike, discountfactor, sigmap, h
# h --> sigmap
# sigmap --> sigma,a,B
# A() --> discountfactors, B, instaforward, a, sigma
# B() --> t,T,a

def B_HW(t,T,a):
    '''
    Help function used in the affine term structure or zero coupon bond pricing.
    :param t: Time of valuation/simulation, usually put on today (0).
    :param T: Time of maturity.
    :param a: Parameter in the HW dynamics.
    :return: B(t,T) function.
    '''
    value = (1 / a) * (1 - np.exp(-a * (T-t)))
    return value

def A_HW(disc_factors,f_vector, t, T,a,sigma):
    '''
    Help function used in the affine term structure or zero coupon bond pricing.
    :param disc_factors: Discount factors (zero-coupon bond prices)
    :param f_vector: Instantaneous forward curve.
    :param t: Time of valuation/simulation, usually put on today (0).
    :param T: Time of maturity.
    :param a: Parameter in the HW dynamics.
    :param sigma: Parameter in the HW dynamics.
    :return: A(t,T) function
    '''
    time_vec = disc_factors[:,0]
    disc_vector = disc_factors[:,1]
    index_t = np.argmin(np.abs(time_vec[:,None] - t),axis=0)
    #index_t = np.where(np.isin(time_vec, t))[0]
    #P(0,t)
    P_M_t = disc_vector[index_t]
    #f(0,t)
    f_M_t = f_vector[index_t]
    index_T = np.argmin(np.abs(time_vec[:,None] - T),axis=0)
    #index_T = np.where(np.isin(time_vec, T))[0]
    #P(0,T)
    P_M_T = disc_vector[index_T]
    #B(t,T)
    B = B_HW(t,T,a)
    value = (P_M_T / P_M_t) * np.exp(B*f_M_t - ((sigma**2 / (4*a)) * (1 - np.exp(-2*a*t)) * (B ** 2)))
    return value

# We can already implement HW_YTM connection: P(t,T) = A(t,T)*exp(-B(t,T)r(t))

def HW_YTM(disc_factors,f_vector,t,T,a,sigma,r_t):
    '''
    Connects the zero-coupon bond pricing with the yield rates under HW model.
    :param disc_factors: Discount factors (zero-coupon bond prices)
    :param f_vector: Instantaneous forward curve.
    :param t: Time of valuation/simulation, usually put on today (0).
    :param T: Time of maturity.
    :param a: Parameter in the HW dynamics.
    :param sigma: Parameter in the HW dynamics.
    :param r_t: Instantaenous spot rate at time t.
    :return: The yield curve for all maturities given in T.
    '''
    A = A_HW(disc_factors,f_vector ,t, T, a, sigma)
    B = B_HW(t,T,a)
    Z = A*np.exp(-B*r_t)
    #Using Annual compounding yield curve.
    Y = (1 / (Z ** (1 / (T-t)))) - 1
    return Y

#Some help functions for using Jamshidian's decomposition for pricing swaptions.
def SigmaP(a,sigma,t,T,S):
    '''
    Calculates help function in pricing formula for HW.
    :param a: Parameter in the HW dynamics.
    :param sigma: Parameter in the HW dynamics.
    :param t: Time of valuation/simulation, usually put on today (0).
    :param T: Time to expiry of option.
    :param S: Time to maturity of underlying.
    :return: sigma_p help constant
    '''
    B = B_HW(T,S,a)
    value = sigma * np.sqrt( (1 - np.exp(-2*a*(T-t))) / (2*a) ) * B
    return value

def h_fun(a,sigma,t,T,S,disc_factors,X):
    '''
    Calculates help function in pricing formula for HW.
    :param a: Parameter in the HW dynamics.
    :param sigma: Parameter in the HW dynamics.
    :param t: Time of valuation/simulation, usually put on today (0).
    :param T: Time to expiry of option.
    :param S: Time to maturity of underlying.
    :param disc_factors: Discount factors (zero-coupon bond prices)
    :param X: Strike of option.
    :return: h help constant
    '''
    #Calculation following p.76 Brigo-Mercurio
    time_vec = disc_factors[:, 0]
    disc_vector = disc_factors[:, 1]
    index_T = np.argmin(np.abs(time_vec[:,None] - T),axis=0)
    index_S = np.argmin(np.abs(time_vec[:,None] - S),axis=0)
    #index_T = np.where(np.isin(time_vec, T))[0]
    #index_S = np.where(np.isin(time_vec, S))[0]
    P_0_T = disc_vector[index_T]
    P_0_S = disc_vector[index_S]
    sigmap = SigmaP(a,sigma,t,T,S)
    value = ((1 / sigmap) * np.log( P_0_S / (P_0_T * X) )) + (sigmap / 2)
    return value

def ZBP(t,T,S,X,a,sigma,disc_factors):
    '''
    Calculates the price of a European put option with strike X, expiry T and written
    on a pure discount bond maturing at time S under HW model.
    :param t: Time of valuation/simulation, usually put on today (0).
    :param T: Time to expiry of option.
    :param S: Time to maturity of underlying.
    :param X: Strike of option.
    :param a: Parameter in the HW dynamics.
    :param sigma: Parameter in the HW dynamics.
    :param disc_factors: Discount factors (zero-coupon bond prices)
    :return: Price of the option, by formula 3.40 in Brigo-Mercurio.
    '''
    time_vec = disc_factors[:, 0]
    disc_vector = disc_factors[:, 1]
    index_T = np.argmin(np.abs(time_vec[:, None] - T), axis=0)
    index_S = np.argmin(np.abs(time_vec[:, None] - S), axis=0)
    #index_T = np.where(np.isin(time_vec, T))[0]
    #index_S = np.where(np.isin(time_vec, S))[0]
    P_0_T = disc_vector[index_T]
    P_0_S = disc_vector[index_S]
    h = h_fun(a,sigma,t,T,S,disc_factors,X)
    sigmap = SigmaP(a,sigma,t,T,S)
    value = (X * P_0_T * norm.cdf(-h + sigmap)) - (P_0_S * norm.cdf(-h))
    return value

def ZBC(t,T,S,X,a,sigma,disc_factors):
    '''
    Calculates the price of a European call option with strike X, expiry T and written
    on a pure discount bond maturing at time S under HW model.
    :param t: Time of valuation/simulation, usually put on today (0).
    :param T: Time to expiry of option.
    :param S: Time to maturity of underlying.
    :param X: Strike of option.
    :param a: Parameter in the HW dynamics.
    :param sigma: Parameter in the HW dynamics.
    :param disc_factors: Discount factors (zero-coupon bond prices)
    :return: Price of the option, by formula 3.40 in Brigo-Mercurio.
    '''
    time_vec = disc_factors[:, 0]
    disc_vector = disc_factors[:, 1]
    index_T = np.argmin(np.abs(time_vec[:, None] - T), axis=0)
    index_S = np.argmin(np.abs(time_vec[:, None] - S), axis=0)
    # index_T = np.where(np.isin(time_vec, T))[0]
    # index_S = np.where(np.isin(time_vec, S))[0]
    P_0_T = disc_vector[index_T]
    P_0_S = disc_vector[index_S]
    #h = h_fun(a, sigma, t, T, S, disc_factors, X)
    #sigmap = SigmaP(a, sigma, t, T, S)
    #value = (P_0_S * norm.cdf(h)) - (X * P_0_T * norm.cdf(h - sigmap))
    zbp = ZBP(t,T,S,X,a,sigma,disc_factors)
    value = zbp + P_0_S - (X*P_0_T)
    return value

def rstar_fun(Strike,Expiry,Maturity,disc_factors,f_vector,a,sigma,r0,data):
    '''
    Calculates value of r* necessary in Jamshidian's option pricing.
    :param Strike: Strike of option
    :param Expiry: Time to expiry of option.
    :param Maturity: Time to maturity of underlying.
    :param disc_factors: Discount factors (zero-coupon bond prices)
    :param f_vector: Instantaneous forward curve.
    :param a: Parameter in the HW dynamics.
    :param sigma: Parameter in the HW dynamics.
    :param r0: The instantaneous spot interest rate observed today.
    :param data: String denoting whether we are dealing with the European dataset or American.
    :return: r* and the value which should be zero.
    '''
    if data == "Europe":
        Tenor = [Expiry + j * 0.5 for j in range(2 * Maturity + 1)]
        ti_vector = np.array(Tenor[1:])
        c_setup = np.zeros(len(ti_vector))
        c = 0.5*Strike + c_setup
        c[-1] = 1 + 0.5*Strike

    if data == "USA":
        Tenor = [Expiry + j for j in range(Maturity + 1)]
        ti_vector = np.array(Tenor[1:])
        c_setup = np.zeros(len(ti_vector))
        c = Strike + c_setup
        c[-1] = 1 + Strike

    A = A_HW(disc_factors,f_vector,Expiry,ti_vector,a,sigma)
    B = B_HW(Expiry,ti_vector,a)
    #We define a help function in terms of r, we search for the r* that makes
    #this function zero.
    def func(r):
        return np.sum(c * A * np.exp(-B * r)) - 1
    initial_guess = r0
    root = fsolve(func, initial_guess)
    zero = func(root)
    return root,zero

def X_vector(Strike,Expiry,Maturity,disc_factors,f_vector,a,sigma,r0,data):
    '''
    Setting up another help function in the pricing formula.
    :param Strike: Strike of option
    :param Expiry: Time to expiry of option.
    :param Maturity: Time to maturity of underlying.
    :param disc_factors: Discount factors (zero-coupon bond prices)
    :param f_vector: Instantaneous forward curve.
    :param a: Parameter in the HW dynamics.
    :param sigma: Parameter in the HW dynamics.
    :param r0: The instantaneous spot interest rate observed today.
    :param data: String denoting whether we are dealing with the European dataset or American.
    :return: X vector used in pricing formula.
    '''
    if data == "Europe":
        Tenor = [Expiry + j * 0.5 for j in range(2 * Maturity + 1)]
    if data == "USA":
        Tenor = [Expiry + j for j in range(Maturity + 1)]
    ti_vector = np.array(Tenor[1:])
    A = A_HW(disc_factors, f_vector, Expiry, ti_vector, a, sigma)
    B = B_HW(Expiry, ti_vector, a)
    r_star,zero = rstar_fun(Strike,Expiry,Maturity,disc_factors,f_vector,a,sigma,r0,data)
    x_vector = A * np.exp(-B * r_star)
    return x_vector

def SwaptionPriceHW(t,Expiry,Maturity,Strike,disc_factors,f_vector,a,sigma,r0,data,Type):
    '''
    Calculates the European swaption price under Hull-White model.
    :param t: Time of valuation/simulation, usually put on today (0).
    :param Expiry: Time to expiry of option.
    :param Maturity: Time to maturity of underlying.
    :param Strike: Strike of option
    :param disc_factors: Discount factors (zero-coupon bond prices)
    :param f_vector: Instantaneous forward curve.
    :param a: Parameter in the HW dynamics.
    :param sigma: Parameter in the HW dynamics.
    :param r0: The instantaneous spot interest rate observed today.
    :param data: String denoting whether we are dealing with the European dataset or American.
    :param Type: Type of the swaption, payer or receiver
    :return: The price of a swaption.
    '''
    #Implements formula 3.45 of Brigo-Mercurio
    if data == "Europe":
        Tenor = [Expiry + j * 0.5 for j in range(2 * Maturity + 1)]
        ti_vector = np.array(Tenor[1:])
        c_setup = np.zeros(len(ti_vector))
        c = 0.5 * Strike + c_setup
        c[-1] = 1 + 0.5 * Strike
        X = X_vector(Strike,Expiry,Maturity,disc_factors,f_vector,a,sigma,r0,data)
        zbp = ZBP(t,Expiry,ti_vector,X,a,sigma,disc_factors)
        price = np.sum(c * zbp)
    if data == "USA":
        Tenor = [Expiry + j for j in range(Maturity + 1)]
        ti_vector = np.array(Tenor[1:])
        c_setup = np.zeros(len(ti_vector))
        c = Strike + c_setup
        c[-1] = 1 + Strike
        X = X_vector(Strike,Expiry,Maturity,disc_factors,f_vector,a,sigma,r0,data)
        zbp = ZBP(t,Expiry,ti_vector,X,a,sigma,disc_factors)
        zbc = ZBC(t,Expiry,ti_vector,X,a,sigma,disc_factors)
        if Type == "P":
            price = np.sum(c * zbp)
        if Type == "R":
            price = np.sum(c * zbc)
        if Type == "Straddle":
            price = np.sum(c * zbp) + np.sum(c * zbc)
    return price

def RMSE_HW(t,Expiries,Maturities,Strikes,disc_factors,f_vector,a,sigma,r0,swaption_prices,data,Type):
    '''
    Objective function to be used in calibration method. Calculates the root mean squared error.
    :param t: Time of valuation/simulation, usually put on today (0).
    :param Expiries: List of time to expiry of options.
    :param Maturities: List of time to maturity of underlyings.
    :param Strikes: List of strike of options
    :param disc_factors: Discount factors (zero-coupon bond prices)
    :param f_vector: Instantaneous forward curve.
    :param a: Parameter in the HW dynamics.
    :param sigma: Parameter in the HW dynamics.
    :param r0: The instantaneous spot interest rate observed today.
    :param swaption_prices: The market swaption prices.
    :param data: String denoting whether we are dealing with the European dataset or American.
    :param Type: List of types of the swaption, payer or receiver
    :return: The root mean squared error of the calibrated prices.
    '''
    N = len(swaption_prices)
    swaptions_HW = np.zeros(N)
    for i in range(N):
        swaptions_HW[i] = SwaptionPriceHW(t,Expiries[i],Maturities[i],Strikes[i],disc_factors,f_vector,a,sigma,r0,data,Type[i])
    rmse = np.sqrt((1 / N) * np.sum((swaption_prices - swaptions_HW) ** 2))
    return rmse

def APE_HW(t,Expiries,Maturities,Strikes,disc_factors,f_vector,a,sigma,r0,swaption_prices,data,Type):
    '''
    Objective function to be used in calibration method. Calculates the average percentage error.
    :param t: Time of valuation/simulation, usually put on today (0).
    :param Expiries: List of time to expiry of options.
    :param Maturities: List of time to maturity of underlyings.
    :param Strikes: List of strike of options
    :param disc_factors: Discount factors (zero-coupon bond prices)
    :param f_vector: Instantaneous forward curve.
    :param a: Parameter in the HW dynamics.
    :param sigma: Parameter in the HW dynamics.
    :param r0: The instantaneous spot interest rate observed today.
    :param swaption_prices: The market swaption prices.
    :param data: String denoting whether we are dealing with the European dataset or American.
    :param Type: List of types of the swaption, payer or receiver
    :return: The average percentage error.
    '''
    N = len(swaption_prices)
    swaptions_HW = np.zeros(N)
    for i in range(N):
        swaptions_HW[i] = SwaptionPriceHW(t, Expiries[i], Maturities[i], Strikes[i], disc_factors, f_vector, a, sigma,r0,data,Type[i])
    ape = (1/ (N * np.mean(swaption_prices))) * np.sum(np.abs(swaption_prices - swaptions_HW))
    return ape

def HW_Calibration(Swaption_file,yield_curve_file,r0,a0,sigma0,objection_function,data):
    '''
    Calibration of the European payer swaption prices under Hull-white model.
    :param Swaption_file: File consisting of the ATM swaption data.
    :param yield_curve_file: File consisting of the yield curve.
    :param r0: The instantaneous spot interest rate observed today.
    :param a0: The starting value for the a parameter.
    :param sigma0: The starting value for the sigma parameter.
    :param objection_function: The objection function to be used in the calibration procedure.
    :param data: String denoting whether we are dealing with the European dataset or American.
    :return: The calibrated parameters of the Hull-White model and the value of the objection function.
    '''
    #Loading in all important data
    disc, _ = DiscountFactors(yield_curve_file,data)
    time_vec = disc[:,0]
    df_fac = disc[:,1]
    f_vector = InstaForward(disc, r0)
    if data == "Europe":
        df_swap = pd.read_excel(Swaption_file, engine='openpyxl')
        Expiries = df_swap["Expiry"].to_numpy()
        Maturities = df_swap["Maturity"].to_numpy()
        Strikes = df_swap["Strike"].to_numpy() / 100
        Types = df_swap["Type"]
        swaption_prices = df_swap["Prices (bp)"].to_numpy() / 10000
    if data == "USA":
        df_swap = pd.read_excel(Swaption_file, engine='openpyxl')
        Expiries = df_swap["Expiry"].to_numpy()
        Maturities = df_swap["Maturity"].to_numpy()
        Strikes = df_swap["Strike"].to_numpy() / 100
        Types = df_swap["Type"]
        swaption_prices_fwd = df_swap["FwdPrice"].to_numpy() / 10000
        indices = np.argmin(np.abs(time_vec[:,None] - Expiries),axis=0)
        swaption_prices = swaption_prices_fwd * df_fac[indices]

    if objection_function == "RMSE":
        obj = lambda x: RMSE_HW(0,Expiries,Maturities,Strikes,disc,f_vector,x[0],x[1],r0,swaption_prices,data,Types)
    if objection_function == "APE":
        obj = lambda x: APE_HW(0,Expiries,Maturities,Strikes,disc,f_vector,x[0],x[1],r0,swaption_prices,data,Types)
    initial_guess = [a0,sigma0]
    #Boundary is of important matter here as the parameters can not be negative.
    bounds = [(0.0001,4),(0.0001,4)]
    result = minimize(obj, initial_guess,bounds=bounds)
    #result = minimize(obj, initial_guess)
    params = result.x
    a = params[0]
    sigma = params[1]
    value = result.fun
    return a,sigma,value

def HW_Prices(Swaption_file,yield_curve_file,r0,a,sigma,data):
    '''
    Returns the prices under the Hull-White model of European payer swaptions.
    :param Swaption_file: File consisting of the ATM swaption data.
    :param yield_curve_file: File consisting of the yield curve.
    :param r0: The instantaneous spot interest rate observed today.
    :param a: Parameter in the HW dynamics.
    :param sigma: Parameter in the HW dynamics.
    :param data: String denoting whether we are dealing with the European dataset or American.
    :return: The swaption prices.
    '''
    disc, _ = DiscountFactors(yield_curve_file, data)
    time_vec = disc[:, 0]
    df_fac = disc[:, 1]
    f_vector = InstaForward(disc, r0)
    if data == "Europe":
        df_swap = pd.read_excel(Swaption_file, engine='openpyxl')
        Expiries = df_swap["Expiry"].to_numpy()
        Maturities = df_swap["Maturity"].to_numpy()
        Strikes = df_swap["Strike"].to_numpy() / 100
        Types = df_swap["Type"]
        swaption_prices = df_swap["Prices (bp)"].to_numpy() / 10000
    if data == "USA":
        df_swap = pd.read_excel(Swaption_file, engine='openpyxl')
        Expiries = df_swap["Expiry"].to_numpy()
        Maturities = df_swap["Maturity"].to_numpy()
        Strikes = df_swap["Strike"].to_numpy() / 100
        Types = df_swap["Type"]
        swaption_prices_fwd = df_swap["FwdPrice"].to_numpy() / 10000
        indices = np.argmin(np.abs(time_vec[:, None] - Expiries), axis=0)
        swaption_prices = swaption_prices_fwd * df_fac[indices]
    N = len(swaption_prices)
    prices = np.zeros(N)
    for i in range(N):
        prices[i] = SwaptionPriceHW(0, Expiries[i], Maturities[i], Strikes[i], disc, f_vector, a, sigma,r0,data,Types[i])
    return prices

#For the simulation of the short rate under Hull-White we will use
# r(t) = x(t) + alpha(t) approach
def alpha_fun(t,a,sigma,time_vec,f_vector):
    '''
    Help function in the simulation procedure of Hull-White short rate.
    :param t: Time of valuation/simulation, usually put on today (0).
    :param a: Parameter in the HW dynamics.
    :param sigma: Parameter in the HW dynamics.
    :param time_vec: Vector with time moments (starting from 3 months till 30 years, with time step one month)
    :param f_vector: Vector with instantaneous forward values.
    :return: alpha(t) used in r(t) = x(t) + alpha(t)
    '''
    index_t = np.argmin(np.abs(time_vec[:, None] - t), axis=0)
    #index_t = np.where(np.isin(time_vec, t))[0]
    f = f_vector[index_t]
    value = f + ((sigma**2) / (2*(a**2))) * ((1 - np.exp(-a*t))**2)
    return value

def simulation_x_process(a,sigma,simulation_time,dt,num_simulations):
    '''
    Simulation of the x(t) in the Hull-White simulation, see 3.38 in Brigo-Mercurio.
    :param a: Parameter in the HW dynamics.
    :param sigma: Parameter in the HW dynamics.
    :param simulation_time: End time of simulation.
    :param dt: Time step in simulation procedure.
    :param num_simulations: Number of simulations.
    :return: Simulated x process at simulation_time.
    '''
    # Number of time steps
    N = int(simulation_time / dt)
    # Make an array to hold every simulated rate for every time step.
    simulated_x = np.zeros((num_simulations, N))
    # Starting rate is known and everywhere the same.
    simulated_x[:, 0] = 0
    # Making the standard normal matrix
    Z = np.random.randn(num_simulations, N - 1)
    for t in range(1,N):
        simulated_x[:, t] = simulated_x[:, t-1] - a*simulated_x[:,t-1]*dt + sigma * np.sqrt(dt) * Z[:, t - 1]
    return simulated_x

def Hull_White_yield_curve_simulation(a,sigma,r0,simulation_time,dt,num_simulations,yield_curve_file,data):
    '''
    Simulation procedure of the yield curve using Hull-White model.
    Under this process, we can only simulate the yield rates for zero-coupon bond
    where the maturity is less than some threshold. We can however, assume a
    constant yield rate for maturities longer than this threshold.
    :param a: Parameter in the HW dynamics.
    :param sigma: Parameter in the HW dynamics.
    :param simulation_time: End time of simulation.
    :param dt: Time step in simulation procedure.
    :param num_simulations: Number of simulations.
    :param yield_curve_file: File consisting of the yield curve.
    :param data: String denoting whether we are dealing with the European dataset or American.
    :return: The simulated yield curve and the maturities.
    '''
    #Number of steps in simulation.
    N = int(simulation_time / dt)
    disc, _ = DiscountFactors(yield_curve_file,data)
    time_vec = disc[:, 0]
    f_vector = InstaForward(disc, r0)
    #We first simulate the x process.
    x = simulation_x_process(a,sigma,simulation_time,dt,num_simulations)
    sim_x = x[:, N-1]
    #We simulate the alpha function.
    sim_alpha = alpha_fun(simulation_time,a,sigma,time_vec,f_vector)
    #The simulated rate is obtained by summing the two previous functions.
    simulated_ro = sim_x + sim_alpha
    df = pd.read_excel(yield_curve_file, engine='openpyxl')
    T_market = df["Maturities"].to_numpy()
    #Last_T = T_market[-1]
    #Last_possible_T = Last_T - simulation_time
    #sub_T_market = T_market[T_market <= Last_possible_T]
    M = len(T_market)
    Result_Matrix = np.zeros((num_simulations, M))
    for i in range(M):
        Result_Matrix[:,i] = HW_YTM(disc,f_vector,simulation_time,T_market[i]+simulation_time,a,sigma,simulated_ro)
    result_percentage = Result_Matrix * 100
    return result_percentage,T_market