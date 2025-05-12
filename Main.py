#Loading in all the packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from NSS_Interpolation import *
from scipy.interpolate import CubicSpline, PchipInterpolator,griddata
from Vasicek import *
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from HW import *
from RTS import *
from MRMClass import *
from scipy.stats import binom
'''
After running this main file, the user is asked which part of the thesis he/she wants to run.
Some chapters have multiple parts and require an extra specification.
All code is self written by the author except for the NSS_Interpolation.py, which was taken from 
https://github.com/open-source-modelling/Nelson_Siegel_Svansson_python the license of this code is
provided as of the other files.
'''
##########################################################################################################
#Chapter 1
def chap_1():
    # Figure of yield curve on 27/11/2024
    file = "Data/YTM_27112024.xlsx"
    df = pd.read_excel(file,engine='openpyxl')
    YTM = df["Yield to maturity"]
    Maturities = df["Maturities"]
    YTM_percentage = YTM*100
    #Parameters obtained from ECB:
    beta0   = 1 # initial guess
    beta1   = 2# initial guess
    beta2   = -0.8# initial guess
    beta3   = 4.5 # initial guess
    lambda0 = 1# initial guess
    lambda1 = 11 # initial guess

    NSS_Maturities = np.linspace(0.25,30,358) #Every month starting from 3 months till 30 years.

    OptiParam = NSSMinimize(beta0, beta1, beta2, beta3, lambda0, lambda1, Maturities, YTM_percentage)
    #NSS Interpolation
    NSS_Yields = NelsonSiegelSvansson(NSS_Maturities, OptiParam[0], OptiParam[1], OptiParam[2], OptiParam[3], OptiParam[4], OptiParam[5])

    plt.plot(NSS_Maturities,NSS_Yields)
    plt.xlabel("Residual maturity in years")
    plt.ylabel("Yield in %")
    plt.xlim(0,30)
    plt.ylim(1.8,2.8)

    plt.show()
##########################################################################################################
#Chapter 4
def chap_4_data():
    # Figure of yield curve on 29 October 2010
    file = "Data/YTM_29102010.xlsx"
    df = pd.read_excel(file,engine='openpyxl')
    YTM = df["Yield to maturity"]
    Maturities = df["Maturities"]
    YTM_percentage = YTM

    beta0   = 2.6 # initial guess
    beta1   = -1.86 # initial guess
    beta2   = 10.64# initial guess
    beta3   = -8 # initial guess
    lambda0 = 5# initial guess
    lambda1 = 3.2 # initial guess

    NSS_Maturities = np.linspace(0.25,30,358) #Every month starting from 3 months till 30 years.

    OptiParam = NSSMinimize(beta0, beta1, beta2, beta3, lambda0, lambda1, Maturities, YTM_percentage)

    NSS_Yields = NelsonSiegelSvansson(NSS_Maturities, OptiParam[0], OptiParam[1], OptiParam[2], OptiParam[3], OptiParam[4], OptiParam[5])

    plt.plot(NSS_Maturities,NSS_Yields)
    plt.xlabel("Residual maturity in years")
    plt.ylabel("Yield in %")
    plt.xlim(0,30)
    plt.ylim(0.5,3.5)

    plt.show()

    # Figure of yield curve on 13/11/2024 U.S. Treasury
    file = "Data/USD/YTM_13NOV2024.xlsx"
    df = pd.read_excel(file, engine='openpyxl')
    YTM = df["Yield to maturity"]
    Maturities = df["Maturities"]

    chip = PchipInterpolator(Maturities, YTM)
    chip_Maturities = np.linspace(1 / 12, 30, 360)
    chip_Yields = chip(chip_Maturities)

    plt.plot(chip_Maturities, chip_Yields)
    plt.xlabel("Residual maturity in years")
    plt.ylabel("Yield in %")
    plt.show()

def chap_4_calibration1():
    #First calibration method and figures
    file_short = "Data/Short_Rates_Historical_Record_29102010.xlsx"
    k0, theta0, sigma0, r0 = Vasicek_Quasi_MLE(file_short, 2000)
    # Taking lambda constant, lambda = 0 as starting point
    file_YTM = "Data/YTM_29102010.xlsx"
    k, theta, sigma, value = Vasicek_Calibration(file_YTM, r0, k0, theta0, sigma0, "APE", "notall")
    print("k","theta","sigma","APE")
    print(k, theta, sigma, value)

    Vasicek_Yield_Curve(file_YTM, r0, k, theta, sigma)

def chap_4_calibration2():
    #Second calibration first two attempts
    file_short = "Data/USD/Short_Rates_13NOV2024.xlsx"
    file_YTM = "Data/USD/Adapted_YTM_13NOV2024.xlsx"

    k0, theta0, sigma0, r0 = Vasicek_Quasi_MLE(file_short, 1500)
    k, theta, sigma, value = Vasicek_Calibration(file_YTM, r0, k0, theta0, sigma0, "APE", "notall")
    print("k", "theta", "sigma", "APE")
    print(k, theta, sigma, value)

    Vasicek_Yield_Curve(file_YTM, r0, k, theta, sigma)

    k0, theta0, sigma0, r0 = Vasicek_Quasi_MLE(file_short, 500)
    k, theta, sigma, value = Vasicek_Calibration(file_YTM, r0, k0, theta0, sigma0, "APE", "notall")
    print("k", "theta", "sigma", "APE")
    print(k, theta, sigma, value)

    Vasicek_Yield_Curve(file_YTM, r0, k, theta, sigma)
    #Second calibration under swaptions.
    file_YTM = "Data/USD/YTM_13NOV2024.xlsx"
    Swaption_file = "Data/USD/filtered_Swaptions.xlsx"

    r0 = 0.0459
    #Found after multiple runnings, provide the best APE
    k0 = 0.031733441270188546
    theta0 = 0.0001
    sigma0 = 0.01077440824671512

    theta, k, sigma, value = Vasicek_Swaption_Calibration(Swaption_file, file_YTM, r0, theta0, k0, sigma0, "USA")
    print("k", "theta", "sigma", "APE")
    print(k, theta, sigma, value)

    #Prices of the datafile are obtained via (*10 000 (in bp))
    #prices_Vas = Vas_Prices(Swaption_file,file_YTM,r0,theta,k,sigma,"USA")
    #Manually add expiry and maturity and market price to obtain the datafile
    #Then we obtain the figure of calibration
    datafile = "Data/USD/USA_ATM_Swaption_Prices_VAS.xlsx"
    df = pd.read_excel(datafile, engine='openpyxl')

    X_unique = np.sort(df["Expiry"].unique())
    Y_unique = np.sort(df["Maturity"].unique())
    X, Y = np.meshgrid(X_unique, Y_unique)
    Z1 = df.pivot_table(index="Maturity", columns="Expiry", values="Prices_M").values
    Z2 = df.pivot_table(index="Maturity", columns="Expiry", values="prices_Vas").values
    Z_diff = np.abs(Z1 - Z2)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 3D surface plot
    ax.plot_surface(X, Y, Z1, color='green', alpha=0.5, edgecolor='k', label='Market')

    # Plot Surface 2 (Blue)
    ax.plot_surface(X, Y, Z2, color='yellow', alpha=0.5, edgecolor='k', label='Vasicek')

    diff_contour = ax.contourf(X, Y, Z_diff, zdir='z', offset=np.min(Z1) - 2, cmap='coolwarm', alpha=0.7)

    fig.colorbar(diff_contour, ax=ax, shrink=0.5, aspect=5, label='Difference Magnitude of Prices (bp)')

    # Add labels
    ax.set_xlabel("Expiry (years)")
    ax.set_ylabel("Tenor (years)")
    ax.set_zlabel("Price (bp)")
    ax.set_title("ATM Swaption Market and Calibrated Vasicek")

    legend_elements = [
        Patch(facecolor='green', label='Market Prices'),
        Patch(facecolor='yellow', label='Vasicek Prices'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    plt.show()

    #Yield curve under the ATM swaption
    file_YTM = "Data/USD/Adapted_YTM_13NOV2024.xlsx"
    Vasicek_Yield_Curve(file_YTM, r0, k, theta, sigma)

    #Last figures, similar as the figure of ATM swaptions but now we include more otm swaptions as well:
    #We present one here, note that for all eight its very similar and you just use the correct datafile
    #The datafile is obtained by determining the prices under the Vasicek model of the parameters
    #obtained in the calibration step.
    datafile = "Data/USD/Calibration_Files_All/-100_VAS.xlsx"
    df = pd.read_excel(datafile, engine='openpyxl')

    X = df["Expiry_-100"].values
    Y = df["Maturity_-100"].values
    Z1 = df["Prices_M_-100"].values
    Z2 = df["Prices_Vas_-100"].values

    X_lin = np.linspace(df["Expiry_-100"].min(), df["Expiry_-100"].max(), 50)
    Y_lin = np.linspace(df["Maturity_-100"].min(), df["Maturity_-100"].max(), 50)
    X_grid, Y_grid = np.meshgrid(X_lin, Y_lin)

    Z1_grid = griddata((X, Y), Z1, (X_grid, Y_grid), method='cubic')
    Z2_grid = griddata((X, Y), Z2, (X_grid, Y_grid), method='cubic')

    Z_diff = np.abs(Z1_grid - Z2_grid)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 3D surface plot
    ax.plot_surface(X_grid, Y_grid, Z1_grid, color='green', alpha=0.5, edgecolor='k', label='Market')

    # Plot Surface 2 (Blue)
    ax.plot_surface(X_grid, Y_grid, Z2_grid, color='yellow', alpha=0.5, edgecolor='k', label='Vasicek')

    diff_contour = ax.contourf(X_grid, Y_grid, Z_diff, zdir='z', offset=np.min(Z1), cmap='coolwarm', alpha=0.7)

    fig.colorbar(diff_contour, ax=ax, shrink=0.5, aspect=5, label='Difference Magnitude of Prices (bp)')

    # Add labels
    ax.set_xlabel("Expiry (years)")
    ax.set_ylabel("Tenor (years)")
    ax.set_zlabel("Price (bp)")
    ax.set_title("-100 bp OTM Swaptions Market and Calibrated Vasicek")

    legend_elements = [
        Patch(facecolor='green', label='Market Prices'),
        Patch(facecolor='yellow', label='Vasicek Prices'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    plt.show()

def chap_4_simulation():
    #Simulated paths example under Vasicek, we take 10 random paths
    X = Vasicek_Sim_Euler(0.272, 0.0389, 0.0167, 0.00724, 3, 1 / 250, 10)
    plt.plot(X.T)
    plt.xlabel("Time steps in number of days")
    plt.ylabel("Short rate r(t)")

    plt.show()

    #Yield curve simulation under Vasicek
    #2010 dataset
    Market_file = "Data/YTM_29102010.xlsx"
    file_short = "Data/Short_Rates_Historical_Record_29102010.xlsx"
    k0, theta0, sigma0, r0 = Vasicek_Quasi_MLE(file_short, 2000)
    # Taking lambda constant, lambda = 0 as starting point
    file_YTM = "Data/YTM_29102010.xlsx"
    k, theta, sigma, value = Vasicek_Calibration(file_YTM, r0, k0, theta0, sigma0, "APE", "notall")
    RHP = 2
    num_simulations = 10000
    Result_Matrix, T_market = Vasicek_Yield_Curve_Simulation(Market_file, file_short, RHP, num_simulations, 2000,
                                                             k, theta, sigma, r0, "test")

    for i in range(30):
        plt.plot(T_market, Result_Matrix[i, :], color="green")

    df = pd.read_excel(Market_file, engine='openpyxl')
    YTM = df["Yield to maturity"]
    plt.plot(T_market, YTM, color="red")
    plt.xlabel("Residual maturity in years")
    plt.ylabel("Yield in %")
    plt.show()

    #2024 dataset
    file_YTM = "Data/USD/YTM_13NOV2024.xlsx"
    Swaption_file = "Data/USD/filtered_Swaptions.xlsx"

    r0 = 0.0459
    # Found after multiple runnings, provide the best APE
    k0 = 0.031733441270188546
    theta0 = 0.0001
    sigma0 = 0.01077440824671512

    theta, k, sigma, value = Vasicek_Swaption_Calibration(Swaption_file, file_YTM, r0, theta0, k0, sigma0, "USA")
    RHP = 2
    num_simulations = 10000
    file_YTM = "Data/USD/Adapted_YTM_13NOV2024.xlsx"
    Result_Matrix, T_market = Vasicek_Yield_Curve_Simulation(file_YTM, file_short, RHP, num_simulations, 2000,
                                                             k, theta, sigma, r0, "test")

    for i in range(30):
        plt.plot(T_market, Result_Matrix[i, :], color="green")

    df = pd.read_excel(file_YTM, engine='openpyxl')
    YTM = df["Yield to maturity"]
    plt.plot(T_market, YTM, color="red")
    plt.xlabel("Residual maturity in years")
    plt.ylabel("Yield in %")
    plt.show()
##########################################################################################################
#Chapter 5
def chap_5_yield_curve():
    #Figure of yield curve
    file = "Data/USD/YTM_13NOV2024.xlsx"
    df = pd.read_excel(file, engine='openpyxl')
    YTM = df["Yield to maturity"]
    Maturities = df["Maturities"]

    chip = PchipInterpolator(Maturities, YTM)
    chip_Maturities = np.linspace(1 / 12, 30, 360)
    chip_Yields = chip(chip_Maturities)

    plt.plot(chip_Maturities, chip_Yields)
    plt.xlabel("Residual maturity in years")
    plt.ylabel("Yield in %")
    plt.show()

    #Figure of yield curve under HW 1:
    file = "Data/USD/Complete_YTM_13NOV2024.xlsx"
    r0 = 0.0459
    df = pd.read_excel(file, engine='openpyxl')
    Mat = df["Maturities"].to_numpy()
    YTM = df["Yield to maturity"] / 100
    N = len(Mat)

    disc, _ = DiscountFactors(file, "USA")
    f_vector = InstaForward(disc, r0)
    Y_HW = HW_YTM(disc, f_vector, 0, Mat, 0.005, 0.2, r0)

    plt.plot(Mat,Y_HW,color="red")
    plt.show()

    #Figure of yield curve under HW 2:
    Y_HW2 = HW_YTM(disc, f_vector, 0, Mat, 0.05, 0.2, r0)
    plt.plot(Mat,Y_HW2,color="red")
    plt.show()
def chap_5_calibration1():
    #Calibrating the parameters
    yield_curve_file = "Data/YTM_29102010.xlsx"
    Swaption_file = "Data/Swaption_Volatilities_2Y_Exp_29102010.xlsx"
    r0 = 0.00724
    a0 = 0.05
    sigma0 = 0.02
    a, sigma, value = HW_Calibration(Swaption_file, yield_curve_file, r0, a0, sigma0, "APE","Europe")
    print("a, sigma, value: ")
    print(a,sigma,value)

    #Making the calibration figure, we have a file that has all prices under hw following 4 different
    #calibration procedures, that is alllowing for pos and negative a under APE and RMSE.
    #We work with APE and pos a as this is the most correct way.
    datafile = "Data/Pricing_Analysis_2Y.xlsx"
    df = pd.read_excel(datafile, engine='openpyxl')

    X_unique = np.sort(df["Expiry"].unique())
    Y_unique = np.sort(df["Maturity"].unique())
    X, Y = np.meshgrid(X_unique, Y_unique)
    Z1 = df.pivot_table(index="Maturity", columns="Expiry", values="P_M (bp)").values
    Z2 = df.pivot_table(index="Maturity", columns="Expiry", values="P_HW (bp) APE pos a").values
    Z_diff = np.abs(Z1 - Z2)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 3D surface plot
    ax.plot_surface(X, Y, Z1, color='green', alpha=0.5, edgecolor='k', label='Market')

    # Plot Surface 2 (Blue)
    ax.plot_surface(X, Y, Z2, color='yellow', alpha=0.5, edgecolor='k', label='Hull-White')

    diff_contour = ax.contourf(X, Y, Z_diff, zdir='z', offset=np.min(Z1) - 2, cmap='coolwarm', alpha=0.7)

    fig.colorbar(diff_contour, ax=ax, shrink=0.5, aspect=5, label='Difference Magnitude of Prices (bp)')

    # Add labels
    ax.set_xlabel("Expiry (years)")
    ax.set_ylabel("Tenor (years)")
    ax.set_zlabel("Price (bp)")
    ax.set_title("ATM Swaption Market and Calibrated HW")

    legend_elements = [
        Patch(facecolor='green', label='Market Prices'),
        Patch(facecolor='yellow', label='Hull-White Prices'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    plt.show()
def chap_5_calibration2():
    #Calibration after iterations
    yield_curve_file = "Data/USD/YTM_13NOV2024.xlsx"

    Swaption_file = "Data/USD/filtered_Swaptions.xlsx"

    r0 = 0.0459
    a0 = 0.03090326746387214
    sigma0 = 0.010770478366555366
    a, sigma, value = HW_Calibration(Swaption_file, yield_curve_file, r0, a0, sigma0, "APE", "USA")
    print("a, sigma, value: ")
    print(a, sigma, value)

    #Creating Figure
    datafile = "Data/USD/USA_ATM_Swaption_Prices.xlsx"
    df = pd.read_excel(datafile, engine='openpyxl')

    X_unique = np.sort(df["Expiry"].unique())
    Y_unique = np.sort(df["Maturity"].unique())
    X, Y = np.meshgrid(X_unique, Y_unique)
    Z1 = df.pivot_table(index="Maturity", columns="Expiry", values="Prices_M").values
    Z2 = df.pivot_table(index="Maturity", columns="Expiry", values="Prices_HW").values
    Z_diff = np.abs(Z1 - Z2)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 3D surface plot
    ax.plot_surface(X, Y, Z1, color='green', alpha=0.5, edgecolor='k', label='Market')

    # Plot Surface 2 (Blue)
    ax.plot_surface(X, Y, Z2, color='yellow', alpha=0.5, edgecolor='k', label='Hull-White')

    diff_contour = ax.contourf(X, Y, Z_diff, zdir='z', offset=np.min(Z1) - 2, cmap='coolwarm', alpha=0.7)

    fig.colorbar(diff_contour, ax=ax, shrink=0.5, aspect=5, label='Difference Magnitude of Prices (bp)')

    # Add labels
    ax.set_xlabel("Expiry (years)")
    ax.set_ylabel("Tenor (years)")
    ax.set_zlabel("Price (bp)")
    ax.set_title("ATM Swaption Market and Calibrated HW")

    legend_elements = [
        Patch(facecolor='green', label='Market Prices'),
        Patch(facecolor='yellow', label='Hull-White Prices'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    plt.show()

    #Including the OTM swaptions:
    df = pd.read_excel("Data/USD/Swaptions_Setup.xlsx")


    filtered_df = df[df["FwdPrice"] > 100]
    filtered_df2 = filtered_df[filtered_df["Expiry"] > 1.5]
    values_to_keep = [0, 25, 50, 75, 100, -25, -50, -75, -100]
    filtered_df3 = filtered_df2[filtered_df2["RelStrike"].isin(values_to_keep)]
    filtered_df4 = filtered_df3[filtered_df3["Maturity"] > 1.5]
    # Save the filtered data to a new Excel file
    filtered_df4.to_excel("Swaptions_Complete.xlsx", index=False)


    file_YTM = "Data/USD/YTM_13NOV2024.xlsx"
    Swaption_file = "Swaptions_Complete.xlsx"

    r0 = 0.0459
    a0 = 0.022531034173237905
    sigma0 = 0.009824952836014738
    a,sigma,value = HW_Calibration(Swaption_file,file_YTM,r0,a0,sigma0,"APE","USA")
    print("a, sigma, value: ")
    print(a,sigma,value)

    #Creating one figure of the 8 of calibrations with OTM swaptions

    datafile = "Data/USD/Calibration_Files_All/-100_HW.xlsx"
    df = pd.read_excel(datafile, engine='openpyxl')

    X = df["Expiry"].values
    Y = df["Maturity"].values
    Z1 = df["Prices_M"].values
    Z2 = df["Prices_HW"].values

    X_lin = np.linspace(df["Expiry"].min(), df["Expiry"].max(), 50)
    Y_lin = np.linspace(df["Maturity"].min(), df["Maturity"].max(), 50)
    X_grid, Y_grid = np.meshgrid(X_lin, Y_lin)

    Z1_grid = griddata((X, Y), Z1, (X_grid, Y_grid), method='cubic')
    Z2_grid = griddata((X, Y), Z2, (X_grid, Y_grid), method='cubic')

    Z_diff = np.abs(Z1_grid - Z2_grid)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 3D surface plot
    ax.plot_surface(X_grid, Y_grid, Z1_grid, color='green', alpha=0.5, edgecolor='k', label='Market')

    # Plot Surface 2 (Blue)
    ax.plot_surface(X_grid, Y_grid, Z2_grid, color='yellow', alpha=0.5, edgecolor='k', label='HW')

    diff_contour = ax.contourf(X_grid, Y_grid, Z_diff, zdir='z', offset=np.min(Z1), cmap='coolwarm', alpha=0.7)

    fig.colorbar(diff_contour, ax=ax, shrink=0.5, aspect=5, label='Difference Magnitude of Prices (bp)')

    # Add labels
    ax.set_xlabel("Expiry (years)")
    ax.set_ylabel("Tenor (years)")
    ax.set_zlabel("Price (bp)")
    ax.set_title("-100 bp OTM Swaptions Market and Calibrated HW")

    legend_elements = [
        Patch(facecolor='green', label='Market Prices'),
        Patch(facecolor='yellow', label='Hull-White Prices'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    plt.show()

def chap_5_simulation():
    #Example of simulated paths under Hull-White
    yield_curve_file = "Data/USD/YTM_13NOV2024.xlsx"
    a = 0.03090326746387214
    sigma = 0.010770478366555366
    r0 = 0.0459

    disc_factors, _ = DiscountFactors(yield_curve_file, "USA")
    f_vector = InstaForward(disc_factors, r0)
    time_vec = disc_factors[:, 0]

    x = simulation_x_process(a, sigma, 3, 1 / 250, 10)
    alpha = alpha_fun(3, a, sigma, time_vec, f_vector)
    sim_r = x + alpha

    plt.plot(sim_r.T)
    plt.xlabel("Time steps in number of days")
    plt.ylabel("Short rate r(t)")

    plt.show()
    #Simulating yield curve under HW
    yield_curve_file = "Data/YTM_29102010.xlsx"

    a = 0.0001
    sigma = 0.0119971013349792
    r0 = 0.00724
    simulation_time = 2
    num_simulations = 10000
    dt = 1 / 250

    Result_Matrix, T_market = Hull_White_yield_curve_simulation(a, sigma, r0, simulation_time, dt, num_simulations,
                                                                yield_curve_file, "USA")
    for i in range(30):
        plt.plot(T_market, Result_Matrix[i, :], color="green")

    df = pd.read_excel(yield_curve_file, engine='openpyxl')
    YTM = df["Yield to maturity"]
    plt.plot(T_market, YTM, color="red")
    plt.xlabel("Residual maturity in years")
    plt.ylabel("Yield in %")
    plt.show()

    #2024 dataset
    yield_curve_file = "Data/USD/Complete_YTM_13NOV2024.xlsx"
    a = 0.03090326746387214
    sigma = 0.010770478366555366
    r0 = 0.0459
    simulation_time = 2
    num_simulations = 10000
    dt = 1 / 250

    Result_Matrix, T_market = Hull_White_yield_curve_simulation(a, sigma, r0, simulation_time, dt, num_simulations,
                                                                yield_curve_file, "USA")
    for i in range(30):
        plt.plot(T_market, Result_Matrix[i, :], color="green")

    df = pd.read_excel(yield_curve_file, engine='openpyxl')
    YTM = df["Yield to maturity"]
    plt.plot(T_market, YTM, color="red")
    plt.xlabel("Residual maturity in years")
    plt.ylabel("Yield in %")
    plt.show()
##########################################################################################################
#Chapter6
def chap_6():
    #Yield curve 2010
    file = "Data/YTM_29102010.xlsx"
    df = pd.read_excel(file, engine='openpyxl')
    YTM = df["Yield to maturity"]
    Maturities = df["Maturities"]
    YTM_percentage = YTM

    beta0 = 2.6  # initial guess
    beta1 = -1.86  # initial guess
    beta2 = 10.64  # initial guess
    beta3 = -8  # initial guess
    lambda0 = 5  # initial guess
    lambda1 = 3.2  # initial guess

    NSS_Maturities = np.linspace(0.25, 30, 358)  # Every month starting from 3 months till 30 years.

    OptiParam = NSSMinimize(beta0, beta1, beta2, beta3, lambda0, lambda1, Maturities, YTM_percentage)

    NSS_Yields = NelsonSiegelSvansson(NSS_Maturities, OptiParam[0], OptiParam[1], OptiParam[2], OptiParam[3],
                                      OptiParam[4], OptiParam[5])

    plt.plot(NSS_Maturities, NSS_Yields)
    plt.xlabel("Residual maturity in years")
    plt.ylabel("Yield in %")
    plt.xlim(0, 30)
    plt.ylim(0.5, 3.5)

    plt.show()
    print("Beta0, Beta1, Beta2, Beta3, lambda0, lambda1")
    print(OptiParam[0], OptiParam[1], OptiParam[2], OptiParam[3],
                                      OptiParam[4], OptiParam[5])

    #Yield curve 2024
    file = "Data/USD/YTM_13NOV2024.xlsx"
    df = pd.read_excel(file, engine='openpyxl')
    YTM = df["Yield to maturity"]
    Maturities = df["Maturities"]

    cs = CubicSpline(Maturities, YTM)
    CS_Maturities = np.linspace(1 / 12, 30, 360)
    CS_Yields = cs(CS_Maturities)

    chip = PchipInterpolator(Maturities, YTM)
    chip_Maturities = np.linspace(1 / 12, 30, 360)
    chip_Yields = chip(chip_Maturities)

    beta0 = 6  # initial guess
    beta1 = -1  # initial guess
    beta2 = -1  # initial guess
    beta3 = 5  # initial guess
    lambda0 = 1  # initial guess
    lambda1 = 11  # initial guess

    NSS_Maturities = np.linspace(1 / 12, 30, 360)  # Every month starting from 3 months till 30 years.

    OptiParam = NSSMinimize(beta0, beta1, beta2, beta3, lambda0, lambda1, Maturities, YTM)

    NSS_Yields = NelsonSiegelSvansson(NSS_Maturities, OptiParam[0], OptiParam[1], OptiParam[2], OptiParam[3],
                                      OptiParam[4], OptiParam[5])

    plt.plot(CS_Maturities, CS_Yields, color="red", label="Cubic")
    plt.scatter(Maturities, YTM, color="blue")
    plt.plot(chip_Maturities, chip_Yields, color="blue", label="Monotone Cubic")
    plt.plot(NSS_Maturities, NSS_Yields, color="yellow", label="NSS")

    plt.xlabel("Residual maturity in years")
    plt.ylabel("Yield in %")
    plt.legend()
    plt.show()
##########################################################################################################
#Chapter 7
def chap_7_procedure():
    #Procedure simulation yield curve 2010
    historical_file = "Data/PRIIPS_RTS_Historical_Record_29102010.xlsx"
    file_current = "Data/YTM_29102010.xlsx"
    RHP = 2
    number_of_sims = 10000
    tenor_points = [0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 28, 29, 30]
    Result, chosen_returns = RTS_Curve_Simulation(historical_file, file_current, RHP, tenor_points, number_of_sims, 0,
                                                  "Europe", False, True)
    for i in range(30):
        plt.plot(tenor_points, Result[i, :], color="green")

    df = pd.read_excel(file_current, engine='openpyxl')
    YTM = df["Yield to maturity"]
    plt.plot(tenor_points, YTM, color="red")
    plt.xlabel("Residual maturity in years")
    plt.ylabel("Yield in %")
    plt.show()

    #Procedure simulation yield curve 2024
    historical_file = "Data/USD/PRIIPS_RTS_Historical_Record_13NOV2024.xlsx"
    file_current = "Data/USD/YTM_13NOV2024.xlsx"
    RHP = 2
    number_of_sims = 10000
    tenor_points = np.array([1 / 12, 1 / 6, 1 / 4, 1 / 2, 1, 2, 3, 5, 7, 10, 20, 30])
    Result, chosen_returns = RTS_Curve_Simulation(historical_file, file_current, RHP, tenor_points, number_of_sims,
                                                  0.01, "USA", False, True)
    for i in range(30):
        chip = PchipInterpolator(tenor_points, Result[i, :])
        chip_Maturities = np.linspace(1 / 12, 30, 360)
        chip_Yields = chip(chip_Maturities)
        plt.plot(chip_Maturities, chip_Yields, color="green")

    df = pd.read_excel(file_current, engine='openpyxl')
    YTM = df["Yield to maturity"]
    chip = PchipInterpolator(tenor_points, YTM)
    chip_Maturities = np.linspace(1 / 12, 30, 360)
    chip_Yields = chip(chip_Maturities)
    plt.plot(chip_Maturities, chip_Yields, color="red")
    plt.xlabel("Residual maturity in years")
    plt.ylabel("Yield in %")
    plt.show()

def chap_7_shift_analyis():
    #For shift of 0.01 %
    historical_file = "Data/USD/PRIIPS_RTS_Historical_Record_13NOV2024.xlsx"
    file_current = "Data/USD/YTM_13NOV2024.xlsx"
    RHP = 3
    number_of_sims = 10000
    tenor_points = np.array([1 / 12, 1 / 6, 1 / 4, 1 / 2, 1, 2, 3, 5, 7, 10, 20, 30])
    Result, chosen_returns = RTS_Curve_Simulation(historical_file, file_current, RHP, tenor_points, number_of_sims,
                                                  0.001, "USA", False, True)
    max_= np.max(Result, axis=0)
    min_ = np.min(Result, axis=0)
    print("Min:", min_)
    print("Max:", max_)
    #For shift of 5 %
    Result, chosen_returns = RTS_Curve_Simulation(historical_file, file_current, RHP, tenor_points, number_of_sims,
                                                  0.05, "USA", False, True)
    max_ = np.max(Result, axis=0)
    min_ = np.min(Result, axis=0)
    print("Min:", min_)
    print("Max:", max_)
    #For shift of 1%
    Result, chosen_returns = RTS_Curve_Simulation(historical_file, file_current, RHP, tenor_points, number_of_sims,
                                                  0.01, "USA", False, True)
    max_ = np.max(Result, axis=0)
    min_ = np.min(Result, axis=0)
    print("Min:", min_)
    print("Max:", max_)

    #Using the zero coupon bond strategy
    historical_file = "Data/USD/PRIIPS_RTS_Historical_Record_13NOV2024.xlsx"
    file_current = "Data/USD/YTM_13NOV2024.xlsx"
    RHP = 3
    number_of_sims = 10000
    tenor_points = np.array([1 / 12, 1 / 6, 1 / 4, 1 / 2, 1, 2, 3, 5, 7, 10, 20, 30])

    # ZCB focus completely negates the shift issue
    zerocouponfile = Zero_Coupon_Bond_Conversion(historical_file, tenor_points)
    Result, chosen_returns = RTS_Curve_Simulation(zerocouponfile, file_current, RHP, tenor_points, number_of_sims, 0,
                                                  "USA", True, True)

    # Descriptive statistics of simulated yields:
    mean = np.mean(Result, axis=0)
    max_ = np.max(Result, axis=0)
    min_ = np.min(Result, axis=0)

    print("Min:", min_)
    print("Max:", max_)
##########################################################################################################
#Chapter 8
def chap_8_FRN_price():
    # 2010
    yield_curve_file = "Data/YTM_29102010.xlsx"
    reference_rate = 0.5
    spread = 0
    maturity = 2
    data = "Europe"
    price,coupon_rates,disc_rates = PriceCalcFRN(yield_curve_file,reference_rate,spread,maturity,data)
    print(price*10000)
    maturity = 3
    price, coupon_rates, disc_rates = PriceCalcFRN(yield_curve_file, reference_rate, spread, maturity, data)
    print(price * 10000)
    maturity = 4
    price, coupon_rates, disc_rates = PriceCalcFRN(yield_curve_file, reference_rate, spread, maturity, data)
    print(price * 10000)
    maturity = 5
    price, coupon_rates, disc_rates = PriceCalcFRN(yield_curve_file, reference_rate, spread, maturity, data)
    print(price * 10000)
    # 2024
    yield_curve_file = "Data/USD/YTM_13NOV2024.xlsx"
    reference_rate = 0.5
    spread = 0
    maturity = 2
    data = "USA"
    price,coupon_rates,disc_rates = PriceCalcFRN(yield_curve_file,reference_rate,spread,maturity,data)
    print(price*10000)
    maturity = 3
    price, coupon_rates, disc_rates = PriceCalcFRN(yield_curve_file, reference_rate, spread, maturity, data)
    print(price * 10000)
    maturity = 4
    price, coupon_rates, disc_rates = PriceCalcFRN(yield_curve_file, reference_rate, spread, maturity, data)
    print(price * 10000)
    maturity = 5
    price, coupon_rates, disc_rates = PriceCalcFRN(yield_curve_file, reference_rate, spread, maturity, data)
    print(price * 10000)

def chap_8_market_risk_FRN():
    # 2010 RTS
    print("2010 RTS")
    file_historical = "Data/PRIIPS_RTS_Historical_Record_29102010.xlsx"
    file_current = "Data/YTM_29102010.xlsx"
    RHP = 2
    spread = 0 / 10000
    number_of_sims = 10000
    data = "Europe"
    tenor_points = np.array([0.25,0.5,0.75,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
    Result,chosen_returns = RTS_Curve_Simulation(file_historical,file_current,RHP,tenor_points,number_of_sims,0,data,False,True)
    VaR_RTS = RTS_VaR_FRN(chosen_returns,file_current,0.5,spread,RHP,0,data,tenor_points,False)
    vev_RTS = VEV_Class(VaR_RTS,RHP)
    print(VaR_RTS*100)
    print(vev_RTS*100)
    RHP = 3
    Result, chosen_returns = RTS_Curve_Simulation(file_historical, file_current, RHP, tenor_points, number_of_sims, 0,
                                                  data, False, True)
    VaR_RTS = RTS_VaR_FRN(chosen_returns, file_current, 0.5, spread, RHP, 0, data, tenor_points, False)
    vev_RTS = VEV_Class(VaR_RTS, RHP)
    print(VaR_RTS * 100)
    print(vev_RTS * 100)
    RHP = 4
    Result, chosen_returns = RTS_Curve_Simulation(file_historical, file_current, RHP, tenor_points, number_of_sims, 0,
                                                  data, False, True)
    VaR_RTS = RTS_VaR_FRN(chosen_returns, file_current, 0.5, spread, RHP, 0, data, tenor_points, False)
    vev_RTS = VEV_Class(VaR_RTS, RHP)
    print(VaR_RTS * 100)
    print(vev_RTS * 100)
    RHP = 5
    Result, chosen_returns = RTS_Curve_Simulation(file_historical, file_current, RHP, tenor_points, number_of_sims, 0,
                                                  data, False, True)
    VaR_RTS = RTS_VaR_FRN(chosen_returns, file_current, 0.5, spread, RHP, 0, data, tenor_points, False)
    vev_RTS = VEV_Class(VaR_RTS, RHP)
    print(VaR_RTS * 100)
    print(vev_RTS * 100)
    # 2010 Vasicek
    print("2010 Vasicek")
    file_current = "Data/YTM_29102010.xlsx"
    RHP = 2
    spread = 0 / 10000
    number_of_sims = 10000
    data = "Europe"
    file_short = "Data/Short_Rates_Historical_Record_29102010.xlsx"
    k0,theta0,sigma0,r0 = Vasicek_Quasi_MLE(file_short,2000)
    #Taking lambda constant, lambda = 0 as starting point
    k,theta,sigma,value = Vasicek_Calibration(file_current,r0,k0,theta0,sigma0,"APE","notall")
    simulated_rates = Vasicek_Sim_Euler(k, theta, sigma, r0, RHP, 1/250, number_of_sims)
    VaR_Vas = Vasicek_VaR_FRN(simulated_rates,file_current,0.5,spread,RHP,k,theta,sigma,data)
    vev_Vas = VEV_Class(VaR_Vas,RHP)
    print(VaR_Vas*100)
    print(vev_Vas*100)
    RHP = 3
    simulated_rates = Vasicek_Sim_Euler(k, theta, sigma, r0, RHP, 1 / 250, number_of_sims)
    VaR_Vas = Vasicek_VaR_FRN(simulated_rates, file_current, 0.5, spread, RHP, k, theta, sigma, data)
    vev_Vas = VEV_Class(VaR_Vas, RHP)
    print(VaR_Vas * 100)
    print(vev_Vas * 100)
    RHP = 4
    simulated_rates = Vasicek_Sim_Euler(k, theta, sigma, r0, RHP, 1 / 250, number_of_sims)
    VaR_Vas = Vasicek_VaR_FRN(simulated_rates, file_current, 0.5, spread, RHP, k, theta, sigma, data)
    vev_Vas = VEV_Class(VaR_Vas, RHP)
    print(VaR_Vas * 100)
    print(vev_Vas * 100)
    RHP = 5
    simulated_rates = Vasicek_Sim_Euler(k, theta, sigma, r0, RHP, 1 / 250, number_of_sims)
    VaR_Vas = Vasicek_VaR_FRN(simulated_rates, file_current, 0.5, spread, RHP, k, theta, sigma, data)
    vev_Vas = VEV_Class(VaR_Vas, RHP)
    print(VaR_Vas * 100)
    print(vev_Vas * 100)

    # 2010 Hull-White
    print("2010 Hull-White")
    file_current = "Data/YTM_29102010.xlsx"
    RHP = 2
    spread = 0 / 10000
    number_of_sims = 10000
    data = "Europe"
    a = 0.0001
    sigma = 0.0119971013349792
    r0 = 0.00724
    simulated_x_rates = simulation_x_process(a,sigma,RHP,1/250,number_of_sims)
    VaR_HW = HW_VaR_FRN(simulated_x_rates,file_current,0.5,spread,RHP,a,sigma,r0,data)
    vev_HW = VEV_Class(VaR_HW,RHP)
    print(VaR_HW*100)
    print(vev_HW*100)
    RHP = 3
    simulated_x_rates = simulation_x_process(a, sigma, RHP, 1 / 250, number_of_sims)
    VaR_HW = HW_VaR_FRN(simulated_x_rates, file_current, 0.5, spread, RHP, a, sigma, r0, data)
    vev_HW = VEV_Class(VaR_HW, RHP)
    print(VaR_HW * 100)
    print(vev_HW * 100)
    RHP = 4
    simulated_x_rates = simulation_x_process(a, sigma, RHP, 1 / 250, number_of_sims)
    VaR_HW = HW_VaR_FRN(simulated_x_rates, file_current, 0.5, spread, RHP, a, sigma, r0, data)
    vev_HW = VEV_Class(VaR_HW, RHP)
    print(VaR_HW * 100)
    print(vev_HW * 100)
    RHP = 5
    simulated_x_rates = simulation_x_process(a, sigma, RHP, 1 / 250, number_of_sims)
    VaR_HW = HW_VaR_FRN(simulated_x_rates, file_current, 0.5, spread, RHP, a, sigma, r0, data)
    vev_HW = VEV_Class(VaR_HW, RHP)
    print(VaR_HW * 100)
    print(vev_HW * 100)

    # Actual ratios and losses 2010
    print("2010 Actual ratios")
    file_current = "Data/YTM_29102010.xlsx"
    RHP = 2
    actual_value,actual_rates = PriceCalcFRNActual(file_current,0.5,0,RHP)
    print(actual_value*100)
    RHP = 3
    actual_value, actual_rates = PriceCalcFRNActual(file_current, 0.5, 0, RHP)
    print(actual_value * 100)
    RHP = 4
    actual_value, actual_rates = PriceCalcFRNActual(file_current, 0.5, 0, RHP)
    print(actual_value * 100)
    RHP = 5
    actual_value, actual_rates = PriceCalcFRNActual(file_current, 0.5, 0, RHP)
    print(actual_value * 100)

    # 2024 RTS
    print("2024 RTS")
    file_historical = "Data/USD/PRIIPS_RTS_Historical_Record_13NOV2024.xlsx"
    file_current = "Data/USD/YTM_13NOV2024.xlsx"
    RHP = 2
    spread = 0 / 10000
    number_of_sims = 10000
    data = "USA"
    tenor_points = np.array([1/12,1/6,1/4,1/2,1,2,3,5,7,10,20,30])
    Result,chosen_returns = RTS_Curve_Simulation(file_historical,file_current,RHP,tenor_points,number_of_sims,0.01,"USA",False,True)
    VaR_RTS = RTS_VaR_FRN(chosen_returns,file_current,0.5,spread,RHP,0.01,"USA",tenor_points,False)
    vev_RTS = VEV_Class(VaR_RTS,RHP)
    print(VaR_RTS*100)
    print(vev_RTS*100)
    RHP = 3
    Result, chosen_returns = RTS_Curve_Simulation(file_historical, file_current, RHP, tenor_points, number_of_sims,
                                                  0.01, "USA", False, True)
    VaR_RTS = RTS_VaR_FRN(chosen_returns, file_current, 0.5, spread, RHP, 0.01, "USA", tenor_points, False)
    vev_RTS = VEV_Class(VaR_RTS, RHP)
    print(VaR_RTS * 100)
    print(vev_RTS * 100)
    RHP = 4
    Result, chosen_returns = RTS_Curve_Simulation(file_historical, file_current, RHP, tenor_points, number_of_sims,
                                                  0.01, "USA", False, True)
    VaR_RTS = RTS_VaR_FRN(chosen_returns, file_current, 0.5, spread, RHP, 0.01, "USA", tenor_points, False)
    vev_RTS = VEV_Class(VaR_RTS, RHP)
    print(VaR_RTS * 100)
    print(vev_RTS * 100)
    RHP = 5
    Result, chosen_returns = RTS_Curve_Simulation(file_historical, file_current, RHP, tenor_points, number_of_sims,
                                                  0.01, "USA", False, True)
    VaR_RTS = RTS_VaR_FRN(chosen_returns, file_current, 0.5, spread, RHP, 0.01, "USA", tenor_points, False)
    vev_RTS = VEV_Class(VaR_RTS, RHP)
    print(VaR_RTS * 100)
    print(vev_RTS * 100)

    # 2024 Vasicek
    print("2024 Vasicek")
    file_current = "Data/USD/YTM_13NOV2024.xlsx"
    RHP = 2
    spread = 0 / 10000
    number_of_sims = 10000
    data = "USA"
    r0 = 0.0459
    theta = 0.0001
    k = 0.031733441270188546
    sigma = 0.01077440824671512
    ape = 0.01617667636878824
    simulated_rates = Vasicek_Sim_Euler(k, theta, sigma, r0, RHP, 1/250, number_of_sims)
    VaR_Vas = Vasicek_VaR_FRN(simulated_rates,file_current,0.5,spread,RHP,k,theta,sigma,data)
    vev_Vas = VEV_Class(VaR_Vas,RHP)
    print(VaR_Vas*100)
    print(vev_Vas*100)
    RHP = 3
    simulated_rates = Vasicek_Sim_Euler(k, theta, sigma, r0, RHP, 1 / 250, number_of_sims)
    VaR_Vas = Vasicek_VaR_FRN(simulated_rates, file_current, 0.5, spread, RHP, k, theta, sigma, data)
    vev_Vas = VEV_Class(VaR_Vas, RHP)
    print(VaR_Vas * 100)
    print(vev_Vas * 100)
    RHP = 4
    simulated_rates = Vasicek_Sim_Euler(k, theta, sigma, r0, RHP, 1 / 250, number_of_sims)
    VaR_Vas = Vasicek_VaR_FRN(simulated_rates, file_current, 0.5, spread, RHP, k, theta, sigma, data)
    vev_Vas = VEV_Class(VaR_Vas, RHP)
    print(VaR_Vas * 100)
    print(vev_Vas * 100)
    RHP = 5
    simulated_rates = Vasicek_Sim_Euler(k, theta, sigma, r0, RHP, 1 / 250, number_of_sims)
    VaR_Vas = Vasicek_VaR_FRN(simulated_rates, file_current, 0.5, spread, RHP, k, theta, sigma, data)
    vev_Vas = VEV_Class(VaR_Vas, RHP)
    print(VaR_Vas * 100)
    print(vev_Vas * 100)

    # 2024 Hull-White
    print("2024 Hull-White")
    file_current = "Data/USD/YTM_13NOV2024.xlsx"
    RHP = 2
    spread = 0 / 10000
    number_of_sims = 10000
    data = "USA"
    a = 0.03090326746387214
    sigma = 0.010770478366555366
    r0 = 0.0459
    simulated_x_rates = simulation_x_process(a,sigma,RHP,1/250,number_of_sims)
    VaR_HW = HW_VaR_FRN(simulated_x_rates,file_current,0.5,spread,RHP,a,sigma,r0,data)
    vev_HW = VEV_Class(VaR_HW,RHP)
    print(VaR_HW*100)
    print(vev_HW*100)
    RHP = 3
    simulated_x_rates = simulation_x_process(a, sigma, RHP, 1 / 250, number_of_sims)
    VaR_HW = HW_VaR_FRN(simulated_x_rates, file_current, 0.5, spread, RHP, a, sigma, r0, data)
    vev_HW = VEV_Class(VaR_HW, RHP)
    print(VaR_HW * 100)
    print(vev_HW * 100)
    RHP = 4
    simulated_x_rates = simulation_x_process(a, sigma, RHP, 1 / 250, number_of_sims)
    VaR_HW = HW_VaR_FRN(simulated_x_rates, file_current, 0.5, spread, RHP, a, sigma, r0, data)
    vev_HW = VEV_Class(VaR_HW, RHP)
    print(VaR_HW * 100)
    print(vev_HW * 100)
    RHP = 5
    simulated_x_rates = simulation_x_process(a, sigma, RHP, 1 / 250, number_of_sims)
    VaR_HW = HW_VaR_FRN(simulated_x_rates, file_current, 0.5, spread, RHP, a, sigma, r0, data)
    vev_HW = VEV_Class(VaR_HW, RHP)
    print(VaR_HW * 100)
    print(vev_HW * 100)

def chap_8_shift_analysis():
    file_historical = "Data/USD/PRIIPS_RTS_Historical_Record_13NOV2024.xlsx"
    file_current = "Data/USD/YTM_13NOV2024.xlsx"
    RHP = 3
    spread = 0 / 10000
    number_of_sims = 10000
    data = "USA"
    tenor_points = np.array([1 / 12, 1 / 6, 1 / 4, 1 / 2, 1, 2, 3, 5, 7, 10, 20, 30])
    shift_vector = np.arange(0.001, 0.0501, 0.001)
    vev_vector = np.zeros(len(shift_vector))
    var_vector = np.zeros(len(shift_vector))
    for i in range(len(shift_vector)):
        Result, chosen_returns = RTS_Curve_Simulation(file_historical, file_current, RHP, tenor_points, number_of_sims,
                                                      shift_vector[i], "USA", False, True)
        VaR_RTS = RTS_VaR_FRN(chosen_returns, file_current, 0.5, spread, RHP, shift_vector[i], "USA", tenor_points,
                              False)
        vev_RTS = VEV_Class(VaR_RTS, RHP)
        vev_vector[i] = vev_RTS * 100
        var_vector[i] = VaR_RTS * 100
    print("Value at risk vector")
    print(var_vector)
    print("VEV vector")
    print(vev_vector)

    print("RTS ZCB method 2024")
    file_historical = "Data/USD/PRIIPS_RTS_Historical_Record_13NOV2024.xlsx"
    file_current = "Data/USD/YTM_13NOV2024.xlsx"
    RHP = 2
    spread = 0 / 10000
    number_of_sims = 10000
    data = "USA"
    tenor_points = np.array([1 / 12, 1 / 6, 1 / 4, 1 / 2, 1, 2, 3, 5, 7, 10, 20, 30])
    zerocouponfile = Zero_Coupon_Bond_Conversion(file_historical, tenor_points)

    Result, chosen_returns = RTS_Curve_Simulation(zerocouponfile, file_current, RHP, tenor_points, number_of_sims, 0,
                                                  "USA", True, True)
    VaR_RTS = RTS_VaR_FRN(chosen_returns, file_current, 0.5, spread, RHP, 0, "USA", tenor_points, True)
    vev_RTS = VEV_Class(VaR_RTS, RHP)
    print(VaR_RTS * 100)
    print(vev_RTS * 100)
    RHP = 3
    Result, chosen_returns = RTS_Curve_Simulation(zerocouponfile, file_current, RHP, tenor_points, number_of_sims, 0,
                                                  "USA", True, True)
    VaR_RTS = RTS_VaR_FRN(chosen_returns, file_current, 0.5, spread, RHP, 0, "USA", tenor_points, True)
    vev_RTS = VEV_Class(VaR_RTS, RHP)
    print(VaR_RTS * 100)
    print(vev_RTS * 100)
    RHP = 4
    Result, chosen_returns = RTS_Curve_Simulation(zerocouponfile, file_current, RHP, tenor_points, number_of_sims, 0,
                                                  "USA", True, True)
    VaR_RTS = RTS_VaR_FRN(chosen_returns, file_current, 0.5, spread, RHP, 0, "USA", tenor_points, True)
    vev_RTS = VEV_Class(VaR_RTS, RHP)
    print(VaR_RTS * 100)
    print(vev_RTS * 100)
    RHP = 5
    Result, chosen_returns = RTS_Curve_Simulation(zerocouponfile, file_current, RHP, tenor_points, number_of_sims, 0,
                                                  "USA", True, True)
    VaR_RTS = RTS_VaR_FRN(chosen_returns, file_current, 0.5, spread, RHP, 0, "USA", tenor_points, True)
    vev_RTS = VEV_Class(VaR_RTS, RHP)
    print(VaR_RTS * 100)
    print(vev_RTS * 100)

def chap_8_tenor_point_analysis():
    historical_file = "Data/PRIIPS_RTS_Historical_Record_29102010.xlsx"
    file_current = "Data/YTM_29102010.xlsx"
    RHP = 2
    number_of_sims = 10000
    spread = 0

    tenor_points = [0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 28, 29, 30]

    first_point = 0.25
    last_point = 30
    semi_point = 0.5
    to_remove = {first_point, last_point, semi_point}
    filtered = [x for x in tenor_points if x not in to_remove]
    act_points = [first_point, last_point, semi_point]
    vev_vector = np.zeros(len(filtered))
    var_vector = np.zeros(len(filtered))
    for i in range(len(filtered)):
        test_vector = np.array(act_points + [filtered[i]])
        Result, chosen_returns = RTS_Curve_Simulation(historical_file, file_current, RHP, test_vector, number_of_sims,
                                                      0, "Europe", False, True)
        VaR_RTS = RTS_VaR_FRN(chosen_returns, file_current, 0.5, spread, RHP, 0, "Europe", test_vector, False)
        vev_RTS = VEV_Class(VaR_RTS, RHP)
        vev_vector[i] = vev_RTS * 100
        var_vector[i] = VaR_RTS * 100
    print("RHP = 2")
    print(np.mean(var_vector))
    print(np.min(var_vector))
    print(np.max(var_vector))
    print(np.mean(vev_vector))
    print(np.min(vev_vector))
    print(np.max(vev_vector))


    RHP = 3
    number_of_sims = 10000
    spread = 0

    tenor_points = [0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 28, 29, 30]

    first_point = 0.25
    last_point = 30
    semi_point = 0.5
    to_remove = {first_point, last_point, semi_point}
    filtered = [x for x in tenor_points if x not in to_remove]
    act_points = [first_point, last_point, semi_point]
    vev_vector = np.zeros(len(filtered))
    var_vector = np.zeros(len(filtered))
    for i in range(len(filtered)):
        test_vector = np.array(act_points + [filtered[i]])
        Result, chosen_returns = RTS_Curve_Simulation(historical_file, file_current, RHP, test_vector, number_of_sims,
                                                      0, "Europe", False, True)
        VaR_RTS = RTS_VaR_FRN(chosen_returns, file_current, 0.5, spread, RHP, 0, "Europe", test_vector, False)
        vev_RTS = VEV_Class(VaR_RTS, RHP)
        vev_vector[i] = vev_RTS * 100
        var_vector[i] = VaR_RTS * 100
    print("RHP = 3")
    print(np.mean(var_vector))
    print(np.min(var_vector))
    print(np.max(var_vector))
    print(np.mean(vev_vector))
    print(np.min(vev_vector))
    print(np.max(vev_vector))

    RHP = 4
    number_of_sims = 10000
    spread = 0

    tenor_points = [0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 28, 29, 30]

    first_point = 0.25
    last_point = 30
    semi_point = 0.5
    to_remove = {first_point, last_point, semi_point}
    filtered = [x for x in tenor_points if x not in to_remove]
    act_points = [first_point, last_point, semi_point]
    vev_vector = np.zeros(len(filtered))
    var_vector = np.zeros(len(filtered))
    for i in range(len(filtered)):
        test_vector = np.array(act_points + [filtered[i]])
        Result, chosen_returns = RTS_Curve_Simulation(historical_file, file_current, RHP, test_vector, number_of_sims,
                                                      0, "Europe", False, True)
        VaR_RTS = RTS_VaR_FRN(chosen_returns, file_current, 0.5, spread, RHP, 0, "Europe", test_vector, False)
        vev_RTS = VEV_Class(VaR_RTS, RHP)
        vev_vector[i] = vev_RTS * 100
        var_vector[i] = VaR_RTS * 100
    print("RHP = 4")
    print(np.mean(var_vector))
    print(np.min(var_vector))
    print(np.max(var_vector))
    print(np.mean(vev_vector))
    print(np.min(vev_vector))
    print(np.max(vev_vector))

    RHP = 5
    number_of_sims = 10000
    spread = 0

    tenor_points = [0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 28, 29, 30]

    first_point = 0.25
    last_point = 30
    semi_point = 0.5
    to_remove = {first_point, last_point, semi_point}
    filtered = [x for x in tenor_points if x not in to_remove]
    act_points = [first_point, last_point, semi_point]
    vev_vector = np.zeros(len(filtered))
    var_vector = np.zeros(len(filtered))
    for i in range(len(filtered)):
        test_vector = np.array(act_points + [filtered[i]])
        Result, chosen_returns = RTS_Curve_Simulation(historical_file, file_current, RHP, test_vector, number_of_sims,
                                                      0, "Europe", False, True)
        VaR_RTS = RTS_VaR_FRN(chosen_returns, file_current, 0.5, spread, RHP, 0, "Europe", test_vector, False)
        vev_RTS = VEV_Class(VaR_RTS, RHP)
        vev_vector[i] = vev_RTS * 100
        var_vector[i] = VaR_RTS * 100
    print("RHP = 5")
    print(np.mean(var_vector))
    print(np.min(var_vector))
    print(np.max(var_vector))
    print(np.mean(vev_vector))
    print(np.min(vev_vector))
    print(np.max(vev_vector))

def chap_8_backtesting():
    r0 = 0.0459
    theta = 0.0001
    k = 0.031733441270188546
    sigma = 0.01077440824671512
    number_of_sims = 1000 #For this amount it takes very long: 40 hours to complete
    #number_of_sims = 2
    simulated_rates = Vasicek_Sim_Euler(k, theta, sigma, r0, 10, 1 / 250, number_of_sims)
    VaR_RTS_vector2 = np.zeros(number_of_sims)
    Actual_value_vector2 = np.zeros(number_of_sims)
    VaR_RTS_vector3 = np.zeros(number_of_sims)
    Actual_value_vector3 = np.zeros(number_of_sims)
    VaR_RTS_vector4 = np.zeros(number_of_sims)
    Actual_value_vector4 = np.zeros(number_of_sims)
    VaR_RTS_vector5 = np.zeros(number_of_sims)
    Actual_value_vector5 = np.zeros(number_of_sims)

    for j in range(number_of_sims):
        rates_for_Vas = np.array([simulated_rates[j, 1250:]])
        rates_for_RTS = simulated_rates[j, :1251]
        rate_for_current = rates_for_RTS[-1]
        tenor_points = np.array(
            [0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
             26, 27, 28, 29, 30])
        yields_for_RTS = np.zeros([len(rates_for_RTS), len(tenor_points)])
        for i in range(len(tenor_points)):
            yields_for_RTS[:, i] = Vasicek_YTM(rates_for_RTS, tenor_points[i], k, theta, sigma) * 100
        yields_for_current = Vasicek_YTM(rate_for_current, tenor_points, k, theta, sigma) * 100
        current_term = np.zeros([len(tenor_points), 2])
        current_term[:, 0] = tenor_points
        current_term[:, 1] = yields_for_current

        term = pd.DataFrame(current_term, columns=["Maturities", "Yield to maturity"])
        term.to_excel("Term_Structure.xlsx", index=False)
        yields = pd.DataFrame(yields_for_RTS, columns=tenor_points)
        yields.to_excel("Yields_RTS.xlsx", index=False)

        min_ = yields.min().min()
        min_shift = 0
        if min_ <= 0:
            min_shift = (-min_ / 100) + 0.01
        # 2 years
        Result, chosen_returns = RTS_Curve_Simulation("Yields_RTS.xlsx", "Term_Structure.xlsx", 2, tenor_points, 10000,
                                                      min_shift, "USA", False, True)
        VaR_RTS = RTS_VaR_FRN(chosen_returns, "Term_Structure.xlsx", 0.5, 0, 2, min_shift, "USA", tenor_points, False)
        VaR_Vas = Vasicek_VaR_FRN(rates_for_Vas, "Term_Structure.xlsx", 0.5, 0, 2, k, theta, sigma, "USA")
        VaR_RTS_vector2[j] = VaR_RTS
        Actual_value_vector2[j] = VaR_Vas[0]
        # 3 years
        Result, chosen_returns = RTS_Curve_Simulation("Yields_RTS.xlsx", "Term_Structure.xlsx", 3, tenor_points, 10000,
                                                      min_shift, "USA", False, True)
        VaR_RTS = RTS_VaR_FRN(chosen_returns, "Term_Structure.xlsx", 0.5, 0, 3, min_shift, "USA", tenor_points, False)
        VaR_Vas = Vasicek_VaR_FRN(rates_for_Vas, "Term_Structure.xlsx", 0.5, 0, 3, k, theta, sigma, "USA")
        VaR_RTS_vector3[j] = VaR_RTS
        Actual_value_vector3[j] = VaR_Vas[0]
        # 4 years
        Result, chosen_returns = RTS_Curve_Simulation("Yields_RTS.xlsx", "Term_Structure.xlsx", 4, tenor_points, 10000,
                                                      min_shift, "USA", False, True)
        VaR_RTS = RTS_VaR_FRN(chosen_returns, "Term_Structure.xlsx", 0.5, 0, 4, min_shift, "USA", tenor_points, False)
        VaR_Vas = Vasicek_VaR_FRN(rates_for_Vas, "Term_Structure.xlsx", 0.5, 0, 4, k, theta, sigma, "USA")
        VaR_RTS_vector4[j] = VaR_RTS
        Actual_value_vector4[j] = VaR_Vas[0]
        # 5 years
        Result, chosen_returns = RTS_Curve_Simulation("Yields_RTS.xlsx", "Term_Structure.xlsx", 5, tenor_points, 10000,
                                                      min_shift, "USA", False, True)
        VaR_RTS = RTS_VaR_FRN(chosen_returns, "Term_Structure.xlsx", 0.5, 0, 5, min_shift, "USA", tenor_points, False)
        VaR_Vas = Vasicek_VaR_FRN(rates_for_Vas, "Term_Structure.xlsx", 0.5, 0, 5, k, theta, sigma, "USA")
        VaR_RTS_vector5[j] = VaR_RTS
        Actual_value_vector5[j] = VaR_Vas[0]

    count2 = np.sum(VaR_RTS_vector2 > Actual_value_vector2)
    print(count2)
    count3 = np.sum(VaR_RTS_vector3 > Actual_value_vector3)
    print(count3)
    count4 = np.sum(VaR_RTS_vector4 > Actual_value_vector4)
    print(count4)
    count5 = np.sum(VaR_RTS_vector5 > Actual_value_vector5)
    print(count5)

    # Bernouilli tests
    prob = 1 - binom.cdf(count2 - 1, number_of_sims, 0.025)
    print(prob * 100)
    prob = 1 - binom.cdf(count3 - 1, number_of_sims, 0.025)
    print(prob * 100)
    prob = 1 - binom.cdf(count4 - 1, number_of_sims, 0.025)
    print(prob * 100)
    prob = 1 - binom.cdf(count5 - 1, number_of_sims, 0.025)
    print(prob * 100)
##########################################################################################################
if __name__ == "__main__":
    choice = input("Which chapter do you want to run? (1/4/5/6/7/8): ")
    if choice == "1":
        print("Code is running...")
        chap_1()
    elif choice == "4":
        choice = input("Which part of this chapter do you want to run? (data/calibration1/calibration2/simulation): ")
        if choice == "data":
            print("Code is running...")
            chap_4_data()
        elif choice == "calibration1":
            print("Code is running...")
            chap_4_calibration1()
        elif choice == "calibration2":
            print("Code is running...")
            chap_4_calibration2()
        elif choice == "simulation":
            print("Code is running...")
            chap_4_simulation()
        else:
            print("Invalid selection")
    elif choice == "5":
        choice = input("Which part of this chapter do you want to run? (yield_curve/calibration1/calibration2/simulation): ")
        if choice == "yield_curve":
            print("Code is running...")
            chap_5_yield_curve()
        elif choice == "calibration1":
            print("Code is running...")
            chap_5_calibration1()
        elif choice == "calibration2":
            print("Code is running...")
            chap_5_calibration2()
        elif choice == "simulation":
            print("Code is running...")
            chap_5_simulation()
        else:
            print("Invalid selection")
    elif choice == "6":
        print("Code is running...")
        chap_6()
    elif choice == "7":
        choice = input("Which part of this chapter do you want to run? (procedure/shift_analysis): ")
        if choice == "procedure":
            print("Code is running...")
            chap_7_procedure()
        elif choice == "shift_analysis":
            print("Code is running...")
            chap_7_shift_analyis()
        else:
            print("Invalid selection")
    elif choice == "8":
        choice = input("Which part of this chapter do you want to run? (FRN_price/market_risk_FRN/shift_analysis/tenor_point_analysis/backtest): ")
        if choice == "FRN_price":
            print("Code is running...")
            chap_8_FRN_price()
        elif choice == "market_risk_FRN":
            print("Code is running...")
            chap_8_market_risk_FRN()
        elif choice == "shift_analysis":
            print("Code is running...")
            chap_8_shift_analysis()
        elif choice == "tenor_point_analysis":
            print("Code is running...")
            chap_8_tenor_point_analysis()
        elif choice == "backtest":
            choice = input("This will take approx. 40 hours to complete, do you want to continue? (Y/N): ")
            if choice == "Y":
                print("Code is running...")
                chap_8_backtesting()
        else:
            print("Invalid selection")
    else:
        print("Invalid selection")






