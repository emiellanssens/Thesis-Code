import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from NSS_Interpolation import *
from HW import DiscountFactors
from scipy.optimize import fsolve
from scipy.stats import norm

#Simulation method via Euler
def Vasicek_Sim_Euler(k, theta, sigma, r0, T, dt, num_simulations):
    """
    Simulating Vasicek model under Euler scheme.
    :param k: Speed of mean reversion level.
    :param theta: Mean reversion level.
    :param sigma: The volatility of the short rate.
    :param r0: The starting rate.
    :param T: Maturity/ End of simulation time.
    :param dt: Step size.
    :param num_simulations: Number of simulated paths
    :return: Simulated interest rates over time.
    """
    #Number of time steps
    N = int(T / dt)
    #Make an array to hold every simulated rate for every time step.
    Rates = np.zeros((num_simulations,N))
    #Starting rate is known and everywhere the same.
    Rates[:, 0] = r0
    #Making the standard normal matrix
    Z = np.random.randn(num_simulations,N-1)
    #Use Euler scheme for discretizing the SDE
    for t in range(1,N):
        Rates[:, t] = Rates[:, t - 1] + k * (theta-Rates[: , t - 1]) * dt + sigma * np.sqrt(dt) * Z[:, t - 1]
    return Rates

#Simulation method via transition density
def Vasicek_Sim_Trans(k, theta, sigma, r0, T, dt, num_simulations):
    """
    Simulating Vasicek model under transition density method.
    :param k: Speed of mean reversion level.
    :param theta: Mean reversion level.
    :param sigma: The volatility of the short rate.
    :param r0: The starting rate.
    :param T: Maturity/ End of simulation time.
    :param dt: Step size.
    :param num_simulations: Number of simulated paths
    :return: Simulated interest rates over time.
    """
    # Number of time steps
    N = int(T / dt)
    # Make an array to hold every simulated rate for every time step.
    Rates = np.zeros((num_simulations, N))
    # Starting rate is known and everywhere the same.
    Rates[:, 0] = r0
    # Making the standard normal matrix
    Z = np.random.randn(num_simulations, N - 1)
    #Use transition density formula for simulation
    alpha = (1 - np.exp(-k * dt))*theta
    beta = np.exp(-k * dt)
    sdd = np.sqrt((sigma**2 / (2*k)) * (1 - np.exp(-2 * k * dt)))
    for t in range(1, N):
        Rates[:, t] = alpha + beta * Rates[:, t-1] + sdd * Z[: , t-1]
    return Rates

#Calibration method via real world --> Quasi Maximum Likelihood Estimator Method
def Vasicek_Quasi_MLE(Sample_Short_Rates_File,last_observations):
    '''
    Calibrate the real-world parameters for the Vasicek model. Note that the
    volatility will remain the same under risk-neutral measure.
    :param Sample_Short_Rates_File: File containing the historical short rates.
    :param last_observations: Number of last observations that will be used.
    :return: The real-world calibrated parameters for the Vasicek model.
    '''
    #Read in the rates and transform them from percentage to decimal
    #We follow the formulas from the calibration paper.
    df_short_rates = pd.read_excel(Sample_Short_Rates_File, engine='openpyxl')
    df = df_short_rates.tail(last_observations) / 100
    a = (df * df.shift(1)).sum()
    b = (df.shift(1)).sum()
    c = (df.shift(-1)).sum()
    n = len(df)
    numerator_beta = a - (1/n)*b*c
    b2 = (df.shift(1) ** 2).sum()
    denom_beta = b2 - (1/n)*(b**2)
    beta = numerator_beta/denom_beta
    alpha = (1/n) * (c - (beta * b))
    d = ((df - alpha - beta * df.shift(1))**2).sum()
    sigma_star_sq = (1/(n-2)) * d
    k = - np.log(beta.iloc[0]) / (1/252)
    theta = (alpha.iloc[0] / (1 - beta.iloc[0]))
    sigma = np.sqrt((2*k*sigma_star_sq.iloc[0]) / (1 - (beta.iloc[0] ** 2)))
    r0_br = df.iloc[-1].to_numpy()
    r0 = r0_br[0]
    return k,theta,sigma,r0

#Calibration method via risk-neutral
def Vasicek_Calibration(Market_file,r0,k0,theta0,sigma,objective_function,method):
    '''
    Calibrate the risk-neutral parameters of the Vasicek model against the current yield curve.
    :param Market_file: Consists of the yield curve observed today.
    :param r0: The instantaneous spot rate observed today.
    :param k0: The starting value of the k parameter (speed of reversion)
    :param theta0: The starting value of the theta parameter (mean reversion level)
    :param sigma: The volatility parameter of Vasicek model.
    :param objective_function: The objection function used in calibration procedure.
    :return: The calibrated parameters and the value of the objection function.
    '''
    df = pd.read_excel(Market_file, engine='openpyxl')
    Y_market = df["Yield to maturity"].to_numpy() / 100
    T_market = df["Maturities"].to_numpy()
    if method != "all":
        if objective_function == "Sum of Squares":
            obj = lambda x : Sum_Squares_Objective(Y_market,T_market,r0,x[0],x[1],sigma)
        if objective_function == "RMSE":
            obj = lambda x : Root_Mean_Squares_Objective(Y_market,T_market,r0,x[0],x[1],sigma)
        if objective_function == "APE":
            obj = lambda x : APE_Vasicek(Y_market,T_market,r0,x[0],x[1],sigma)
        initial_guess = [k0,theta0]
        bounds = [[0.0001,5],[0.0001,5]]
        result = minimize(obj,initial_guess,bounds=bounds)
        params = result.x
        k = params[0]
        theta = params[1]
    if method == "all":
        if objective_function == "Sum of Squares":
            obj = lambda x : Sum_Squares_Objective(Y_market,T_market,r0,x[0],x[1],x[2])
        if objective_function == "RMSE":
            obj = lambda x : Root_Mean_Squares_Objective(Y_market,T_market,r0,x[0],x[1],x[2])
        if objective_function == "APE":
            obj = lambda x : APE_Vasicek(Y_market,T_market,r0,x[0],x[1],x[2])
        initial_guess = [k0,theta0,sigma]
        bounds = [[0.0001, 5], [0.0001, 5],[0.0001, 5]]
        result = minimize(obj, initial_guess, bounds=bounds)
        params = result.x
        k = params[0]
        theta = params[1]
        sigma = params[2]
    value = result.fun
    return k,theta,sigma,value

#Creating the sum of squares minimization
def Sum_Squares_Objective(Y_market,T_market,r0,k,theta,sigma):
    '''
    Objective function used in calibration procedure.
    :param Y_market: The observed yield curve
    :param T_market: The maturities used in the observed yield curve
    :param r0: The instantaneous spot rate observed today.
    :param k: Parameter of Vasicek model.
    :param theta: Parameter of Vasicek model.
    :param sigma: Parameter of Vasicek model.
    :return: Sum of squared error.
    '''
    N = len(Y_market)
    Y_vasicek = np.zeros(N)
    for i in range(N):
        Y_vasicek[i] = Vasicek_YTM(r0,T_market[i],k,theta,sigma)
    value = np.sum((Y_market - Y_vasicek)**2)
    return value

def Root_Mean_Squares_Objective(Y_market,T_market,r0,k,theta,sigma):
    '''
    Objective function to be used in calibration method. Calculates the root mean squared error.
    :param Y_market: The observed yield curve
    :param T_market: The maturities used in the observed yield curve
    :param r0: The instantaneous spot rate observed today.
    :param k: Parameter of Vasicek model.
    :param theta: Parameter of Vasicek model.
    :param sigma: Parameter of Vasicek model.
    :return: The root mean squared error of the calibrated zero coupon bonds.
    '''
    N = len(Y_market)
    Y_vasicek = np.zeros(N)
    for i in range(N):
        Y_vasicek[i] = Vasicek_YTM(r0, T_market[i], k, theta, sigma)
    value = np.sqrt( (1/N) * np.sum((Y_market - Y_vasicek) ** 2))
    return value

def APE_Vasicek(Y_market,T_market,r0,k,theta,sigma):
    '''
    Objective function to be used in calibration method. Calculates the average percentage error.
    :param Y_market: The observed yield curve
    :param T_market: The maturities used in the observed yield curve
    :param r0: The instantaneous spot rate observed today.
    :param k: Parameter of Vasicek model.
    :param theta: Parameter of Vasicek model.
    :param sigma: Parameter of Vasicek model.
    :return: The average percentage error.
    '''
    N = len(Y_market)
    Y_vasicek = np.zeros(N)
    for i in range(N):
        Y_vasicek[i] = Vasicek_YTM(r0, T_market[i], k, theta, sigma)
    value = (1/ (N * np.mean(Y_market))) * np.sum(np.abs(Y_market - Y_vasicek))
    return value


#Connection YTM <--> short rate
def Vasicek_YTM(r0,T,k,theta,sigma):
    '''
    Connects the zero-coupon bond pricing with the yield rates under Vasicek model.
    :param r0: The instantaneous spot rate observed today.
    :param T: The maturity of the zero-coupon bond
    :param k: Parameter of Vasicek model
    :param theta: Parameter of Vasicek model
    :param sigma: Parameter of Vasicek model
    :return: The yield curve for all maturities given in T.
    '''
    B = (1 / k) * (1 - np.exp(-k * T))
    A = (B - T) * (theta - (sigma ** 2 / (2 * k ** 2))) - sigma ** 2 * B ** 2 / (4*k)
    Z = np.exp(A - (B * r0))
    Y = (1 / (Z ** (1/T))) - 1
    return Y

def Vasicek_Yield_Curve_Simulation(Market_file,Historical_file,RHP,num_simulations,last_observations,k,theta,sigma,r0,method):
    '''
    Simulates the yield curve under Vasicek model.
    :param Market_file: Consists of the yield curve observed today.
    :param Historical_file: File containing the historical short rates.
    :param RHP: Recommended holding period, or simulation time.
    :param num_simulations: Number of simulations.
    :param last_observations: Number of observations included of the historical file.
    :return: The simulated yield rates.
    '''
    #Start by calibrating the real world parameters.
    if method == "MLE":
        k0, theta0, sigma0, r0 = Vasicek_Quasi_MLE(Historical_file, last_observations)
    #We then calibrate the risk neutral parameters, where the sigma parameter
    #was obtained from the MLE method.
        k, theta, sigma, value = Vasicek_Calibration(Market_file, r0, k0, theta0, sigma0, "APE","notall")
    #We simulate the short rate
    Rates = Vasicek_Sim_Euler(k, theta, sigma, r0, RHP, 1/252, num_simulations)
    N = int(RHP / (1/252))
    Simulated_r0s = Rates[: , N - 1]
    df = pd.read_excel(Market_file, engine='openpyxl')
    T_market = df["Maturities"].to_numpy()
    M = len(T_market)
    Result_Matrix = np.zeros((num_simulations,M))
    #For every maturity, we calculate the corresponding yield rate under Vasicek model.
    for i in range(M):
        Result_Matrix[:,i] = Vasicek_YTM(Simulated_r0s,T_market[i],k,theta,sigma)
    result_percentage = Result_Matrix * 100
    return result_percentage,T_market

def Vasicek_Yield_Curve(Market_file,r0,k,theta,sigma):
    '''
    This function plots the yield curve under the Vasicek model.
    :param Market_file: File containing the yield curve observed today.
    :param r0: The instantaneous spot rate of today.
    :param k: Parameter of the Vasicek model.
    :param theta: Parameter of the Vasicek model.
    :param sigma: Parameter of the Vasicek model.
    :return: None
    '''
    #Read in the market file with the yield curve
    df = pd.read_excel(Market_file, engine='openpyxl')
    YTM = df["Yield to maturity"]
    Maturities = df["Maturities"]
    YTM_percentage = YTM

    T_market = Maturities.to_numpy()
    #Create the yield curve under the Vasicek model
    Y_vasicek = Vasicek_YTM(r0, T_market, k, theta, sigma)

    Y_vasi_percentage = Y_vasicek * 100
    #Plot both the Vasicek yield curve and the observed yield curve
    plt.plot(T_market, YTM_percentage,label="Market", color="blue")
    plt.plot(T_market, Y_vasi_percentage,label="Vasicek", color="red")

    plt.xlabel("Residual maturity in years")
    plt.ylabel("Yield in %")
    plt.xlim(0, 30)
    plt.legend()
    plt.show()
    return None

def r_star_fun_Vas(Strike,Expiry,Maturity,theta,k,sigma,r0,data):
    '''
    Function that finds the r* used in the pricing of swaptions.
    :param Strike: The strike of the swaption.
    :param Expiry: The expiry of the option.
    :param Maturity: The maturity of the underlying swap.
    :param theta: Parameter of the Vasicek model.
    :param k: Parameter of the Vasicek model.
    :param sigma: Parameter of the Vasicek model.
    :param r0: The instantaneous spot rate observed today.
    :param data: String denoting whether we are dealing with the European dataset or the American.
    :return: r* and the zero value of the function.
    '''
    #Depending on the dataset we set up the payment times (tenor) of the underlying swap
    #For the European dataset we assume semi-annual payments
    if data == "Europe":
        Tenor = [Expiry + j * 0.5 for j in range(2 * Maturity + 1)]
        ti_vector = np.array(Tenor[1:])
        c_setup = np.zeros(len(ti_vector))
        c = 0.5*Strike + c_setup
        c[-1] = 1 + 0.5*Strike
    #For the American dataset we have annual payments
    if data == "USA":
        Tenor = [Expiry + j for j in range(Maturity + 1)]
        ti_vector = np.array(Tenor[1:])
        c_setup = np.zeros(len(ti_vector))
        c = Strike + c_setup
        c[-1] = 1 + Strike
    #Implementing the formula for the A and B functions under the Vasicek model
    B = (1 / k) * (1 - np.exp(-k * (ti_vector-Expiry)))
    A = np.exp( ((B - ti_vector + Expiry) * (theta - (sigma ** 2 / (2 * k ** 2)))) - (sigma ** 2 * B ** 2 / (4 * k)))
    #Find the r* that equals to 1 (see derivation in Chapter of Hull-White)
    def func(r):
        return np.sum(c * A * np.exp(-B * r)) - 1
    initial_guess = r0
    root = fsolve(func, initial_guess)
    zero = func(root)
    return root,zero

def X_vector_Vas(Strike,Expiry,Maturity,theta,k,sigma,r0,data):
    '''
    This function creates the vector of strikes used in the pricing of the swaptions, using
    the coupon bond option pricing formula.
    :param Strike: The strike of the swaption.
    :param Expiry: The expiry of the option.
    :param Maturity: The maturity of the underlying swap.
    :param theta: Parameter of the Vasicek model.
    :param k: Parameter of the Vasicek model.
    :param sigma: Parameter of the Vasicek model.
    :param r0: The instantaneous spot rate observed today.
    :param data: String denoting whether we are dealing with the European or American dataset.
    :return: The vector of strikes x.
    '''
    #Making the distinction between European and American data
    if data == "Europe":
        Tenor = [Expiry + j * 0.5 for j in range(2 * Maturity + 1)]
    if data == "USA":
        Tenor = [Expiry + j for j in range(Maturity + 1)]

    #We follow the derived formula.
    ti_vector = np.array(Tenor[1:])

    B = (1 / k) * (1 - np.exp(-k * (ti_vector - Expiry)))
    A = np.exp(((B - ti_vector + Expiry) * (theta - (sigma ** 2 / (2 * k ** 2)))) - (sigma ** 2 * B ** 2 / (4 * k)))
    r_star, zero = r_star_fun_Vas(Strike,Expiry,Maturity,theta,k,sigma,r0,data)
    x_vector = A * np.exp(-B * r_star)

    return x_vector

def ZBO_Vas(Strike,T,S,theta,k,sigma,type,disc_factors):
    '''
    Creates the price of an option on a zero-coupon bond by implementing the analytical formula under
    the Vasicek model.
    :param Strike: The strike of the option.
    :param T: The expiry of the option.
    :param S: The maturity of the underlying zero-coupon bond.
    :param theta: Parameter of the Vasicek model.
    :param k: Parameter of the Vasicek model.
    :param sigma: Parameter of the Vasicek model.
    :param type: String denoting whether we are dealing with a call or put option.
    :param disc_factors: The discountfactors derived from the observed yield curve of today.
    :return: The price of the zero-coupon bond option
    '''
    #Making a distinction between a call and a put option
    if type == "call":
        w = 1
    if type == "put":
        w = -1
    #Obtain the time vector and the discount factors
    time_vec = disc_factors[:, 0]
    disc_vector = disc_factors[:, 1]
    index_T = np.argmin(np.abs(time_vec[:, None] - T), axis=0)
    index_S = np.argmin(np.abs(time_vec[:, None] - S), axis=0)
    #Compute the price using the analytical expression under the Vasicek model
    P_0_T = disc_vector[index_T]
    P_0_S = disc_vector[index_S]
    B = (1 / k) * (1 - np.exp(-k * (S - T)))
    sigmap = sigma*np.sqrt( (1 - np.exp(-2*k*T)) / (2*k)  ) * B
    h = (1 / sigmap) * np.log( P_0_S / (P_0_T*Strike) ) + (sigmap/2)
    zbo = w * ((P_0_S * norm.cdf(w*h))  - (Strike*P_0_T*norm.cdf(w*(h - sigmap))) )
    return zbo

def Swaption_Pricing_Vasicek(Strike,Expiry,Maturity,theta,k,sigma,r0,disc_factors,data,Type):
    '''
    Comput the price of a swaption under the Vasicek model.
    :param Strike: The strike of the swaption.
    :param Expiry: The expiry of the option.
    :param Maturity: The maturity of the underlying swap.
    :param theta: Parameter of the Vasicek model.
    :param k: Parameter of the Vasicek model.
    :param sigma: Parameter of the Vasicek model.
    :param r0: The instantaneous spot rate observed today.
    :param disc_factors: The discountfactors derived from the observed yield curve of today.
    :param data: String denoting whether we are dealing with the American or European dataset.
    :param Type: String denoting whether we hava a Straddle, Payer or Receiver swaption.
    :return: The price of the swaption.
    '''
    #Make a distinction between the European and American dataset
    #Assume semi-annual payments for the European, while we have annual payments for the American.
    #Implement the formula from the Jamshidian decomposition.
    if data == "Europe":
        Tenor = [Expiry + j * 0.5 for j in range(2 * Maturity + 1)]
        ti_vector = np.array(Tenor[1:])
        c_setup = np.zeros(len(ti_vector))
        c = 0.5 * Strike + c_setup
        c[-1] = 1 + 0.5 * Strike
        X = X_vector_Vas(Strike,Expiry,Maturity,theta,k,sigma,r0,data)
        zbp = ZBO_Vas(X,Expiry,ti_vector,theta, k, sigma, "put", disc_factors)
        price = np.sum(c * zbp)
    if data == "USA":
        Tenor = [Expiry + j for j in range(Maturity + 1)]
        ti_vector = np.array(Tenor[1:])
        c_setup = np.zeros(len(ti_vector))
        c = Strike + c_setup
        c[-1] = 1 + Strike
        X = X_vector_Vas(Strike,Expiry,Maturity,theta,k,sigma,r0,data)
        zbp = ZBO_Vas(X,Expiry,ti_vector,theta, k, sigma, "put", disc_factors)
        zbc = ZBO_Vas(X,Expiry,ti_vector,theta, k, sigma, "call", disc_factors)
        if Type == "P":
            price = np.sum(c * zbp)
        if Type == "R":
            price = np.sum(c * zbc)
        if Type == "Straddle":
            price = np.sum(c * zbp) + np.sum(c * zbc)
    return price

def APE_Swaptions_Vasicek(Expiries,Maturities,Strikes,disc_factors,theta,k,sigma,r0,swaption_prices,data,Type):
    '''
    Calculates the average percentage error between market prices of swaptions and the Vasicek model prices.
    :param Expiries: List of expiries of the swaptions.
    :param Maturities: List of maturities of the underlying swaps.
    :param Strikes: List of the strikes of the swaptions.
    :param disc_factors: The discountfactors derived from the observed yield curve.
    :param theta: Parameter of the Vasicek model.
    :param k: Parameter of the Vasicek model.
    :param sigma: Parameter of the Vasicek model.
    :param r0: The instantaneous spot rate observed today.
    :param swaption_prices: The market prices of the swaptions.
    :param data: String denoting whether we are dealing with the American or European dataset.
    :param Type: List of types of the swaptions.
    :return: The absolute percentage error
    '''
    N = len(swaption_prices)
    swaptions_Vas = np.zeros(N)
    for i in range(N):
        swaptions_Vas[i] = Swaption_Pricing_Vasicek(Strikes[i],Expiries[i],Maturities[i],theta,k,sigma,r0,disc_factors,data,Type[i])
    ape = (1 / (N * np.mean(swaption_prices))) * np.sum(np.abs(swaption_prices - swaptions_Vas))
    return ape

def Vasicek_Swaption_Calibration(Swaption_file,yield_curve_file,r0,theta0,k0,sigma0,data):
    '''
    Function of the calibration procedure for swaptions under the Vasicek model.
    :param Swaption_file: Market file consisting of swaptions.
    :param yield_curve_file: File containing the observed yield curve.
    :param r0: The instantaneous spot rate observed today.
    :param theta0: Starting parameter of the Vasicek model.
    :param k0: Starting parameter of the Vasicek model.
    :param sigma0: Starting parameter of the Vasicek model.
    :param data: String denoting whether we are dealing with the American or European dataset.
    :return: The calibrated parameters of the Vasicek model.
    '''
    #Obtain the discount factors from the yield curve.
    disc, _ = DiscountFactors(yield_curve_file, data)
    time_vec = disc[:, 0]
    df_fac = disc[:, 1]
    #Make a distinction between the European and American dataset as the latter
    #denotes the pricing in a forward price.
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
    #Use the APE objective function
    obj = lambda x: APE_Swaptions_Vasicek(Expiries,Maturities,Strikes,disc,x[0],x[1],x[2],r0,swaption_prices,data,Types)
    initial_guess = [theta0,k0,sigma0]
    bounds = [[0.0001,5],[0.0001,5],[0.0001,5]]
    result = minimize(obj, initial_guess, bounds=bounds)
    params = result.x
    theta = params[0]
    k = params[1]
    sigma = params[2]
    value = result.fun
    return theta,k,sigma,value

def Vas_Prices(Swaption_file,yield_curve_file,r0,theta,k,sigma,data):
    '''
    This function calculates the prices of the swaptions under the calibrated Vasicek parameters.
    :param Swaption_file: Market file that contains the swaptions.
    :param yield_curve_file: File that contains the observed yield curve.
    :param r0: The instantaneous spot rate observed today.
    :param theta: The calibrated parameter of the Vasicek model.
    :param k: The calibrated parameter of the Vasicek model.
    :param sigma: The calibrated parameter of the Vasicek model.
    :param data: String denoting whether we are dealing with the American or European dataset.
    :return: A list of prices of the swaptions.
    '''
    #Obtain the discountfactors from the yield curve
    disc, _ = DiscountFactors(yield_curve_file, data)
    time_vec = disc[:, 0]
    df_fac = disc[:, 1]
    #Make a distinction between the American and European datasets.
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
    #List the prices under the calibrated parameters of the Vasicek model
    N = len(swaption_prices)
    prices = np.zeros(N)
    for i in range(N):
        prices[i] = Swaption_Pricing_Vasicek(Strikes[i],Expiries[i],Maturities[i],theta,k,sigma,r0,disc,data,Types[i])
    return prices










