# Marcelo Leszynski 
# 03/15/22
# MATH 405 Sec 002

from scipy.stats.distributions import norm
from scipy.optimize import fmin, minimize
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pydataset import data as pydata
from statsmodels.tsa.stattools import arma_order_select_ic as order_select
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VARMAX
from statsmodels.tsa.base.datetools import dates_from_str

def arma_forecast_naive(file='weather.npy', p=2,q=1,n=20):
    """
    Perform ARMA(1,1) on data. Let error terms be drawn from
    a standard normal and let all constants be 1.
    Predict n values and plot original data with predictions.

    Parameters:
        file (str): data file
        p (int): order of autoregressive model
        q (int): order of moving average model
        n (int): number of future predictions
    """
    # initialize values ########################################################
    temps = np.load(file)
    d_z = np.diff(temps)
    phi = 0.5
    theta = 0.1
    eps = [norm().rvs()]
    preds = [temps[-i] for i in range(1, p + 1)]

    for t in range(n):
        c = 0
        epsilon = norm.rvs()
        AR = np.sum([phi*preds[-i] for i in range(1, p + 1)])
        MA = np.sum([theta*eps[-j] for j in range(1, q + 1)])
        eps.append(epsilon)
        preds.append(c + AR + MA + epsilon)

    # plot results #############################################################
    d_1 = [13 + (20 + i)/24 for i in range(len(d_z))]
    d_2 = [16 + (19 + i)/24 for i in range(n)]

    plt.plot(d_1, d_z, label='Old Data')
    plt.plot(d_2, np.diff(preds[1:]), label='New Data')
    plt.title('ARMA(2,1) Naive Forecast')
    plt.xlabel('Day of the Month')
    plt.ylabel('Change in Temperature $(C)-\mu = 0$')
    plt.legend()
    plt.show()
        

def arma_likelihood(file='weather.npy', phis=np.array([0.9]), thetas=np.array([0]), mu=17., std=0.4):
    """
    Transfer the ARMA model into state space. 
    Return the log-likelihood of the ARMA model.

    Parameters:
        file (str): data file
        phis (ndarray): coefficients of autoregressive model
        thetas (ndarray): coefficients of moving average model
        mu (float): mean of errorm
        std (float): standard deviation of error

    Return:
        log_likelihood (float)
    """
    # initialize values ########################################################
    temps = np.load(file)
    d_z = np.diff(temps)
    series = d_z - mu
    F, Q, H, dim_states, dim_time_series = state_space_rep(phis, thetas, mu, std)
    means, covs = kalman(F, Q, H, series)
    logs = []

    # calculate state space ####################################################
    for i in range(len(series)):
        mean = H@means[i]
        cov = np.sqrt(H@covs[i]@H.T)
        logs.append(np.log(norm(loc=mean, scale=cov).pdf(series[i])))

    log_likelihood = np.sum(logs)

    return log_likelihood


def model_identification(file='weather.npy',p=4,q=4):
    """
    Identify parameters to minimize AIC of ARMA(p,q) model

    Parameters:
        file (str): data file
        p (int): maximum order of autoregressive model
        q (int): maximum order of moving average model

    Returns:
        phis (ndarray (p,)): coefficients for AR(p)
        thetas (ndarray (q,)): coefficients for MA(q)
        mu (float): mean of error
        std (float): std of error
    """

    # initialize values ########################################################
    temps = np.load(file)
    t_series = np.diff(temps)
    n = len(t_series)
    AIC = np.inf

    # use spec code to identify parameters for AIC #############################
    for i in range(1, p+1):
        for j in range(1, q+1):
            def f(x): # x contains the phis, thetas, mu, and std
                return -1*arma_likelihood(file, phis=x[:i], thetas=x[i:i+j], mu=x[-2], std=x[-1])
            # create initial point
            x0 = np.zeros(i+j+2)
            x0[-2] = t_series.mean()
            x0[-1] = t_series.std()
            sol = minimize(f, x0, method = "SLSQP")
            opt = sol["fun"]
            sol = sol['x']

            k = i + j + 2
            temp_AIC = 2*k*(1 + (k + 1)/(n - k)) - 2*opt

            if temp_AIC < AIC:
                AIC = temp_AIC
                phis = sol[:i]
                thetas = sol[i:i + j]
                mu = sol[-2]
                std = sol[-1]

    return phis, thetas, mu, std
    

def arma_forecast(file='weather.npy', phis=np.array([0]), thetas=np.array([0]), mu=0., std=0., n=30):
    """
    Forecast future observations of data.
    
    Parameters:
        file (str): data file
        phis (ndarray (p,)): coefficients of AR(p)
        thetas (ndarray (q,)): coefficients of MA(q)
        mu (float): mean of ARMA model
        std (float): standard deviation of ARMA model
        n (int): number of forecast observations

    Returns:
        new_mus (ndarray (n,)): future means
        new_covs (ndarray (n,)): future standard deviations
    """
    # initialize values ########################################################
    temps = np.load(file)
    t_series = np.diff(temps)
    #length=len(time_series)
    new_mus = []
    new_covs = []
    F, Q, H, dim_states, dim_time_series = state_space_rep(phis, thetas, mu, std)
    means, covs = kalman(F, Q, H, t_series)
    x_k_1 = means[-1]
    P_k_1 = covs[-1]

    # perform first update #####################################################
    y = t_series[-1] - H@x_k_1
    S = H@P_k_1@H.T
    K = P_k_1@H.T@np.linalg.inv(S)
    x_kk = x_k_1 + K@y
    P_kk = P_k_1 - K@H@P_k_1

    # perform all other updates ################################################
    for i in range(n):
        x_k = F@x_kk + mu
        P_k = F@P_kk@F.T + Q
        new_mus.append(np.mean(H@x_k + mu))
        new_covs.append(np.mean(H@P_k@H.T))
        x_kk = x_k
        P_kk = P_k

    cov_upp = new_mus + 2*np.sqrt(new_covs)
    cov_low = new_mus - 2*np.sqrt(new_covs)

    # plot results #############################################################
    d_1 = [13 + (20+i)/24 for i in range(len(t_series))]
    d_2 = [16 + (19+i)/24 for i in range(n)]

    plt.plot(d_1, t_series, label='Old Data')
    plt.plot(d_2, new_mus, label='Forecast')
    plt.plot(d_2, cov_upp, color='green', label='95% Confidence Interval')
    plt.plot(d_2, cov_low, color='green')
    plt.xlabel('Day of the Month')
    plt.ylabel('Change in Temperature $(C)-\mu=0$')
    k = (n+18) // 24
    plt.locator_params(axis='x', nbins=4 + k)
    plt.title('Problem 4 ARMA(1,1)')
    plt.legend()
    plt.show()

    return new_mus, new_covs


def sm_arma(file='weather.npy', p=4, q=4, n=30):
    """
    Build an ARMA model with statsmodel and 
    predict future n values.

    Parameters:
        file (str): data file
        p (int): maximum order of autoregressive model
        q (int): maximum order of moving average model
        n (int): number of values to predict

    Return:
        aic (float): aic of optimal model
    """
    # initialize values ########################################################
    temps = np.load(file)
    t_series = np.diff(temps)
    length=len(t_series)

    # build model ##############################################################
    aic = np.inf
    for i in range(1, p+1):
        # make predictions #####################################################
        for j in range(1, q+1):
            model = ARIMA(t_series,order=(i,0,j),trend='c').fit(method='innovations_mle')
            temp_AIC = model.aic
            if temp_AIC < aic:
                aic = temp_AIC
                predics = model.predict(start=0, end=length+n)
    
    # plot results #############################################################
    d_1 = [13 + (20 + i)/24 for i in range(len(t_series))]
    d_2 = [13 + (20 + i)/24 for i in range(len(predics) - 1)]

    plt.plot(d_1, t_series, label='Old Data')
    plt.plot(d_2, np.diff(predics), label='ARMA Model')
    plt.ylabel('Change in Temperature $(C)-\mu=0$')
    plt.xlabel('Day of the Month')
    plt.title('Statsmodel ARMA(1,1)')
    plt.legend()
    plt.show()

    return aic


def sm_varma(start='1959-09-30', end='2012-09-30'):
    """
    Build an ARMA model with statsmodel and
    predict future n values.

    Parameters:
        start (str): the data at which to begin forecasting
        end (str): the date at which to stop forecasting

    Return:
        aic (float): aic of optimal model
    """
    # Load in data
    df = sm.datasets.macrodata.load_pandas().data
    # Create DateTimeIndex
    dates = df[['year', 'quarter']].astype(int).astype(str)
    dates = dates["year"] + "Q" + dates["quarter"]
    dates = dates_from_str(dates)
    df.index = pd.DatetimeIndex(dates)
    # Select columns used in prediction
    df = df[['realgdp','realcons','realinv']]
    
    # initialize values and make predictions ###################################
    mod = VARMAX(df)
    mod = mod.fit(maxiter=1000, disp=False, ic = 'aic')
    pred = mod.predict(start, end)

    # create confidence intervals ##############################################
    forecast_obj = mod.get_forecast(end)
    all_CI = forecast_obj.conf_int(alpha=0.05)

    # plot results in subplots #################################################
    fig, ax = plt.subplots(3,1, figsize=(15,8))
    ax[0].plot(df['realgdp'], color='blue', label='realgdp')
    ax[0].plot(pred['realgdp'], color='orange', label='forecast')
    ax[0].plot(all_CI['lower realgdp'], 'k--', label='95% confidence interval')
    ax[0].plot(all_CI['upper realgdp'], 'k--')
    ax[0].set_title('realgdp prediction')
    ax[0].set_xlabel('year')
    ax[0].set_ylabel('realgdp')
    ax[0].legend()
    ax[1].plot(df['realcons'], color='blue', label='realcons')
    ax[1].plot(pred['realcons'], color='orange', label='forecast')
    ax[1].plot(all_CI['lower realcons'], 'k--', label='95% confidence interval')
    ax[1].plot(all_CI['upper realcons'], 'k--')
    ax[1].set_title('realcons prediction')
    ax[1].set_xlabel('year')
    ax[1].set_ylabel('realcons')
    ax[1].legend()
    ax[2].plot(df['realinv'], color='blue', label='realinv')
    ax[2].plot(pred['realinv'], color='orange', label='forecast')
    ax[2].plot(all_CI['lower realinv'], 'k--', label='95% confidence interval')
    ax[2].plot(all_CI['upper realinv'], 'k--')
    ax[2].set_title('realinv prediction')
    ax[2].set_xlabel('year')
    ax[2].set_ylabel('realinv')
    ax[2].legend()
    fig.suptitle('Problem 6')
    fig.tight_layout()
    plt.show()

    aic = mod.aic

    return aic


def manaus(start='1983-01-31',end='1995-01-31',p=4,q=4):
    """
    Plot the ARMA(p,q) model of the River Negro height
    data using statsmodels built-in ARMA class.

    Parameters:
        start (str): the data at which to begin forecasting
        end (str): the date at which to stop forecasting
        p (int): max_ar parameter
        q (int): max_ma parameter
    Return:
        aic_min_order (tuple): optimal order based on AIC
        bic_min_order (tuple): optimal order based on BIC
    """
    # Get dataset
    raw = pydata('manaus')
    # Make DateTimeIndex
    manaus = pd.DataFrame(raw.values,index=pd.date_range('1903-01','1993-01',freq='M'))
    manaus = manaus.drop(0,axis=1)
    # Reset column names
    manaus.columns = ['Water Level']

    raise NotImplementedError("Problem 7 Incomplete")

###############################################################################
    
def kalman(F, Q, H, time_series):
    # Get dimensions
    dim_states = F.shape[0]

    # Initialize variables
    # covs[i] = P_{i | i-1}
    covs = np.zeros((len(time_series), dim_states, dim_states))
    mus = np.zeros((len(time_series), dim_states))

    # Solve of for first mu and cov
    covs[0] = np.linalg.solve(np.eye(dim_states**2) - np.kron(F,F),np.eye(dim_states**2)).dot(Q.flatten()).reshape(
            (dim_states,dim_states))
    mus[0] = np.zeros((dim_states,))

    # Update Kalman Filter
    for i in range(1, len(time_series)):
        t1 = np.linalg.solve(H.dot(covs[i-1]).dot(H.T),np.eye(H.shape[0]))
        t2 = covs[i-1].dot(H.T.dot(t1.dot(H.dot(covs[i-1]))))
        covs[i] = F.dot((covs[i-1] - t2).dot(F.T)) + Q
        mus[i] = F.dot(mus[i-1]) + F.dot(covs[i-1].dot(H.T.dot(t1))).dot(
                time_series[i-1] - H.dot(mus[i-1]))
    return mus, covs

def state_space_rep(phis, thetas, mu, sigma):
    # Initialize variables
    dim_states = max(len(phis), len(thetas)+1)
    dim_time_series = 1 #hardcoded for 1d time_series

    F = np.zeros((dim_states,dim_states))
    Q = np.zeros((dim_states, dim_states))
    H = np.zeros((dim_time_series, dim_states))

    # Create F
    F[0][:len(phis)] = phis
    F[1:,:-1] = np.eye(dim_states - 1)
    # Create Q
    Q[0][0] = sigma**2
    # Create H
    H[0][0] = 1.
    H[0][1:len(thetas)+1] = thetas

    return F, Q, H, dim_states, dim_time_series


# test code ####################################################################
# if __name__ == "__main__":
    # arma_forecast_naive()
    # print(arma_likelihood())
    # print(model_identification())
    # phis, thetas, mu, std = np.array([0.72135856]), np.array([-0.26246788]), 0.35980339870105321, 1.5568331253098422
    # arma_forecast(file='weather.npy', phis=phis, thetas=thetas, mu=mu, std=std)
    # sm_arma(p=3, q=3, n=30)
    # sm_varma()