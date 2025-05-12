import numpy as np
from HW import *
from Vasicek import *

def PriceCalcFRN(yield_curve_file,reference_rate,spread,maturity,data):
    '''
    The function calculates the price of a floating rate note where the variable
    rate is given by the reference rate (0.5 --> yield rate on a 6M ZCB, 0.25 --> yield rate on a 3M ZCB)
    with spread and maturity. We follow standard pricing procedure where the coupon rates
    are determined by the expectation hypothesis based on the observed yield curve.
    :param yield_curve_file: File consisting of the yield curve observed today.
    :param reference_rate: 0.25/0.5 denoting 6M or 3M yield rate as variable rate.
    :param spread: A number e.g. 20 bp, denoting the constant spread above or under the variable rate.
    :param maturity: The maturity of the FRN in years.
    :param data: String denoting whether we are dealing with the European dataset or American.
    :return: The price of the Floating rate note, present value of all future cash flows.
    '''
    #We need the yields and discountfactors
    discountfactors, yields = DiscountFactors(yield_curve_file,data)
    #reference rate must be provided in years and < 1 (0.25/0.5)
    #also denotes the frequency of payments of the FRN
    time_vec = discountfactors[:,0]
    ZCB = discountfactors[:,1]
    number_of_payments = int(maturity/reference_rate)
    coupon_rates = np.zeros(number_of_payments)
    disc_rates = np.zeros(number_of_payments)
    index_reference_today = np.argmin(np.abs(time_vec[:,None] - reference_rate),axis=0)
    #index_reference_today = np.where(np.isin(time_vec, reference_rate))[0]
    obs_reference_rate = yields[index_reference_today-1]
    coupon_rates[0] = obs_reference_rate
    #We create the coupon rates using the expectation hypothesis and
    #the discountfactors by interpolating the payment dates.
    for i in range(1,number_of_payments):
        index_reference_i_1 = np.argmin(np.abs(time_vec[:,None] - ((i+1)*reference_rate)),axis=0)
        #index_reference_i_1 = np.where(np.isin(time_vec, (i+1)*reference_rate))[0]
        yield_i_1 = yields[index_reference_i_1-1]
        index_reference_i = np.argmin(np.abs(time_vec[:,None] - (i*reference_rate)),axis=0)
        #index_reference_i = np.where(np.isin(time_vec, i*reference_rate))[0]
        yield_i = yields[index_reference_i-1]
        coupon_rates[i] = ((((1+yield_i_1)**((i+1)*reference_rate)) / ((1+yield_i) ** (i*reference_rate))) ** (1 / reference_rate)) - 1
    for j in range(number_of_payments):
        index_payment = np.argmin(np.abs(time_vec[:,None] - ((j+1)*reference_rate)),axis=0)
        #index_payment = np.where(np.isin(time_vec, (j+1)*reference_rate))[0]
        disc_rates[j] = ZCB[index_payment]
    #We have an unit notional thus price is achieved by summing the expected cash flows and discounting.
    #We added the notional on the end (last discountfactor, notional is 1)
    price = np.sum((coupon_rates + spread) * reference_rate * disc_rates) + disc_rates[-1]
    return price,coupon_rates,disc_rates

def RTS_VaR_FRN(chosen_returns,yield_curve_file,reference_rate,spread,maturity,shift,data,tenor_points,methodzcb):
    '''
    This function determines the Value at Risk following the RTS procedure of a floating rate note. We
    determine the price you invest in the bond and rank the ratio's of price invested and price gained
    where we simulate under the exact procedure. The value at risk is obtained by selecting the 97.5% lowest
    ratio.
    :param chosen_returns: These are the returns following the RTS procedure (shifted for positive values).
    :param yield_curve_file: The file that contains the observed yield curve.
    :param reference_rate: The rate on which the floating rate note is based.
    :param spread: The added spread on the reference rate for coupons.
    :param maturity: The maturity of the bond.
    :param shift: The shift that was used in creating positive yields.
    :param data: String denoting whether we are dealing with US or European data.
    :param tenor_points: The tenor points that make up the yield curve.
    :param methodzcb: Boolean denoting whether we are working on the zero-coupon bond prices or not.
    :return: The value at risk under the RTS procedure.
    '''
    #We determine the price of the floating rate notes, while also
    #having the expected rates and used discount rates.
    price_invest,expected_rates,disc_rates = PriceCalcFRN(yield_curve_file,reference_rate,spread,maturity,data)
    #Number of times payments occur under the floating rate note
    number_of_payments = len(expected_rates)
    tenor_point_index = np.argmin(np.abs(tenor_points[:,None] - reference_rate),axis=0)[0]
    #We make sure we select the correct yield of maturities (equal the reference rate)
    returns_reference_rate = chosen_returns[:,:,tenor_point_index]
    number_of_simulations = len(returns_reference_rate[:,0])
    #Select the correct times in the complete simulations, the first one is obtained after total / number of payments.
    index_simulation_moments = int(len(returns_reference_rate[0,:])/number_of_payments)
    #We will determine the coupon rates per simulated path
    coupon_rates = np.zeros([number_of_simulations,number_of_payments])
    #The first coupon rate is already known via the expected rate
    coupon_rates[:,0] = expected_rates[0]
    for i in range(1,number_of_payments):
        #Select the correct simulation and shift for positive value and shift to have mean
        #equal to expected rate
        sim_stopping_point_index = (i*index_simulation_moments)
        sample_return_stopping_point = returns_reference_rate[:,:sim_stopping_point_index]
        sum_sampled_returns = np.sum(sample_return_stopping_point,axis=1)
        if methodzcb == False:
            sim_rate_unshifted = (coupon_rates[:,0]*np.exp(sum_sampled_returns)) - shift
        if methodzcb == True:
            zcb_today = 1 / ((1 + coupon_rates[:,0])**reference_rate)
            sim_zcb = zcb_today*np.exp(sum_sampled_returns)
            sim_rate_unshifted = ((1 / sim_zcb)**(1 / reference_rate))  - 1
    current_mean = np.mean(sim_rate_unshifted)
    sim_shifted_rate = sim_rate_unshifted - current_mean + expected_rates[i]
    coupon_rates[:, i] = sim_shifted_rate
    #Discountfactors remain the same.
    disc_matrix = np.tile(disc_rates, (number_of_simulations,1))
    #Determine the value of the floating rate note using the simulated rates + discount until the
    #start as we compare this value with the start price.
    price_gained = np.sum((coupon_rates + spread) * reference_rate * disc_matrix, axis=1) + disc_matrix[:,-1]
    #Compute the ratio
    all_ratios = price_gained / price_invest
    VaR_Rank = int(0.975*number_of_simulations)
    ranks = np.argsort(-all_ratios).argsort() + 1
    #Determine the Value at risk.
    VaR = all_ratios[ranks == VaR_Rank][0]
    return VaR

def VEV_Class(VaR,RHP):
    '''
    This function determines the Value at risk equivalent volatility using the formula of the RTS.
    :param VaR: The computed Value at Risk.
    :param RHP: The recommended holding period of the instrument (Maturity for FRN).
    :return: VEV
    '''
    VEV = (np.sqrt(3.842 - (2*np.log(VaR))) - 1.96) / np.sqrt(RHP)
    return VEV

def Vasicek_VaR_FRN(simulated_rates,yield_curve_file,reference_rate,spread,maturity,k,theta,sigma,data):
    '''
    This function calculates the Value at risk following the RTS procedure but using the simulated yield
    curve under the Vasicek model.
    :param simulated_rates: The simulated short rates r(t) under Vasicek until maturity of the bond.
    :param yield_curve_file: The file containing the observed yield curve.
    :param reference_rate: The reference rate used for the floating rate note.
    :param spread: The spread added on the reference rate for the coupons of the floating rate note.
    :param maturity: The maturity of the bond.
    :param k: Parameter of the Vasicek model.
    :param theta: Parameter of the Vasicek model.
    :param sigma: Parameter of the Vasicek model.
    :param data: String denoting whether we are dealing with the US or European dataset.
    :return: The value at risk.
    '''
    #The procedure runs similar as RTS_VaR_FRN but now using the simulated rates of the Vasicek model.
    price_invest,expected_rates,disc_rates = PriceCalcFRN(yield_curve_file,reference_rate,spread,maturity,data)
    number_of_payments = len(expected_rates)
    number_of_simulations = len(simulated_rates[:,0])
    index_simulation_moments = int(len(simulated_rates[0,:])/number_of_payments)
    coupon_rates = np.zeros([number_of_simulations, number_of_payments])
    coupon_rates[:, 0] = expected_rates[0]
    for i in range(1,number_of_payments):
        #We make sure we select the correct simulated rate at every payment time
        sim_stopping_point_index = (i*index_simulation_moments) - 1
        simulated_ro = simulated_rates[:,sim_stopping_point_index]
        sim_rates = Vasicek_YTM(simulated_ro,reference_rate,k,theta,sigma)
        #Create the yield to maturity from the simulated rate.
        coupon_rates[:,i] = sim_rates
    disc_matrix = np.tile(disc_rates, (number_of_simulations, 1))
    price_gained = np.sum((coupon_rates + spread) * reference_rate * disc_matrix, axis=1) + disc_matrix[:, -1]
    all_ratios = price_gained / price_invest
    if number_of_simulations == 1:
        VaR = all_ratios
    else:
        VaR_Rank = int(0.975 * number_of_simulations)
        ranks = np.argsort(-all_ratios).argsort() + 1
        VaR = all_ratios[ranks == VaR_Rank][0]
    return VaR

def HW_VaR_FRN(simulated_x_rates,yield_curve_file,reference_rate,spread,maturity,a,sigma,r0,data):
    '''
    This function calculates the Value at risk following the RTS procedure but using the simulated yield
    curve under the Hull-White model.
    :param simulated_x_rates: The simulated x-process used in creating the simulated short rate.
    :param yield_curve_file: The file containing the observed yield curve.
    :param reference_rate: The reference rate used for the floating rate note.
    :param spread: The spread added on the reference rate for the coupons of the floating rate note.
    :param maturity: The maturity of the bond.
    :param a: Parameter of the Hull-White model.
    :param sigma: Parameter of the Hull-White model.
    :param r0: The observed short rate at time 0.
    :param data: String denoting whether we are dealing with the US or European dataset.
    :return: The Value at risk.
    '''
    #The procedure runs similar as RTS_VaR_FRN but now using the simulated rates of the Hull-White model.
    disc, _ = DiscountFactors(yield_curve_file,data)
    time_vec = disc[:, 0]
    f_vector = InstaForward(disc, r0)
    price_invest, expected_rates, disc_rates = PriceCalcFRN(yield_curve_file, reference_rate, spread, maturity,data)
    number_of_payments = len(expected_rates)
    number_of_simulations = len(simulated_x_rates[:, 0])
    index_simulation_moments = int(len(simulated_x_rates[0, :]) / number_of_payments)
    coupon_rates = np.zeros([number_of_simulations, number_of_payments])
    coupon_rates[:, 0] = expected_rates[0]
    for i in range(1,number_of_payments):
        #Find the correct simulated short rates using the x-process and the alpha function.
        sim_stopping_point_index = (i*index_simulation_moments) - 1
        simulated_x = simulated_x_rates[:,sim_stopping_point_index]
        simulated_alpha = alpha_fun(i*reference_rate,a,sigma,time_vec,f_vector)
        simulated_ro = simulated_x + simulated_alpha
        #Use the yield rate via the Hull-White formula for yield rate and short rate.
        sim_rates = HW_YTM(disc,f_vector,i*reference_rate,(i+1)*reference_rate,a,sigma,simulated_ro)
        coupon_rates[:, i] = sim_rates
    disc_matrix = np.tile(disc_rates, (number_of_simulations, 1))
    price_gained = np.sum((coupon_rates + spread) * reference_rate * disc_matrix, axis=1) + disc_matrix[:, -1]
    all_ratios = price_gained / price_invest
    VaR_Rank = int(0.975 * number_of_simulations)
    ranks = np.argsort(-all_ratios).argsort() + 1
    VaR = all_ratios[ranks == VaR_Rank][0]
    return VaR

def PriceCalcFRNActual(yield_curve_file,reference_rate,spread,maturity):
    '''
    This function calculates the actual ratio of value of the FRN over the price it costs to invest in.
    This function is specifically for the 2010 dataset as there we know how the actual yield curve
    will evolve.
    :param yield_curve_file: The file containing the observed yield curve.
    :param reference_rate: The reference rate used for the floating rate note.
    :param spread: The spread added on the reference rate for the coupons of the floating rate note.
    :param maturity: The maturity of the bond.
    :return: The actual ratio and the coupon rates.
    '''
    price_invest,expected_rates,disc_rates = PriceCalcFRN(yield_curve_file,reference_rate,spread,maturity,"Europe")
    #The actual rates can be made into input...
    coupon_rates_per = np.array([0.769410,	1.157741,	0.424226,	0.043174,	-0.000418,	0.004698,0.051253,	0.105372,-0.042696,	-0.287823])
    coupon_rates = coupon_rates_per[:int(maturity/reference_rate)] / 100
    price = np.sum((coupon_rates + spread) * reference_rate * disc_rates) + disc_rates[-1]
    actual_ratio = price / price_invest
    return actual_ratio,coupon_rates







