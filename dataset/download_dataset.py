import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np
import talib as tb
import cv2
'''
DEFINING SOME VARIABLES
'''
startDate = '2001-10-11'
endDate = '2022-04-15'
axes = ['Date', 'Value']
headers = ['RSI', 'CMO', 'PLUS_DI', 'MINUS_DI', 'WILLR', 'CCI', 'ULTOSC', 'AROONOSC', 'MFI', 'MOM', 'MACD', 'MACDFIX', 'LINEARREG_ANGLE', 'LINEARREG_SLOPE', 'ROCP', 'ROC', 'ROCR', 'ROCR100', 'SLOWK',
           'FASTD', 'SLOWD', 'AROONUP', 'AROONDOWN', 'APO', 'MACDEXT', 'FASTK', 'PPO', 'MINUS_DM', 'ADOSC', 'FASTDRSI', 'FASTKRSI', 'TRANGE', 'TRIX', 'STD', 'BOP', 'VAR', 'PLUS_DM', 'CORREL', 'AD',
           'BETA', 'WCLPRICE', 'TSF', 'TYPPRICE', 'AVGPRICE', 'MEDPRICE', 'BBANDSL', 'LINEARREG', 'OBV', 'BBANDSM', 'TEMA', 'BBANDSU', 'DEMA', 'MIDPRICE', 'MIDPOINT', 'WMA', 'EMA',
           'HT_TRENDLINE', 'KAMA', 'SMA', 'MA', 'ADXR', 'ADX', 'TRIMA', 'LINEARREG_INTERCEPT', 'DX']


etfList = ['XLF', 'XLU', 'QQQ', 'SPY', 'XLP', 'EWZ', 'EWH', 'XLY', 'XLE']
threshold = 0.01  # Re-arrange the Threshold Value

pd.set_option('display.max_rows', None)

'''
DOWNLOADING THE DATA
'''
# DataFrame, size=(n_days, 6), col_names=["Open", "High", "Low", "Close", "Adj Close", "Volume"]
for etf in etfList:

    imageList = []
    labelList = []

    data = yf.download(etf, start=startDate, end=endDate, )
    '''
    CALCULATING THE INDICATOR VALUES
    '''
    # DataFrame, size=(n_days, 2), col_names=["Date", "Value"]
    rsi = tb.RSI(data["Close"], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    wma = tb.WMA(data["Close"], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    ema = tb.EMA(data["Close"], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    sma = tb.SMA(data["Close"], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    roc = tb.ROC(data["Close"], timeperiod=10).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    cmo = tb.CMO(data["Close"], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    cci = tb.CCI(data["High"], data["Low"], data["Close"],
                 timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    ppo = tb.PPO(data["Close"], fastperiod=12, slowperiod=26,
                 matype=0).to_frame().reset_index().set_axis(axes, axis=1)
    tema = tb.TEMA(data["Close"], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    willr = tb.WILLR(data["High"], data["Low"], data["Close"],
                     timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    macd, macdsignal, macdhist = tb.MACD(
        data["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
    macd = macd.to_frame().reset_index().set_axis(axes, axis=1)

    sar = tb.SAR(data["High"], data["Low"], acceleration=0,
                 maximum=0).to_frame().reset_index().set_axis(axes, axis=1)
    adx = tb.ADX(data["High"], data["Low"], data["Close"],
                 timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    std = tb.STDDEV(data['Close'], timeperiod=5, nbdev=1).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    obv = tb.OBV(data['Close'], data['Volume']).to_frame(
    ).reset_index().set_axis(axes, axis=1)

    adxr = tb.ADXR(data["High"], data["Low"], data["Close"],
                   timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    apo = tb.APO(data['Close'], fastperiod=12, slowperiod=26,
                 matype=0).to_frame().reset_index().set_axis(axes, axis=1)
    aroondown, aroonup = tb.AROON(data["High"], data["Low"], timeperiod=14)
    aroondown = aroondown.to_frame().reset_index().set_axis(axes, axis=1)
    aroonup = aroonup.to_frame().reset_index().set_axis(axes, axis=1)
    aroonosc = tb.AROONOSC(data["High"], data["Low"], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    bop = tb.BOP(data["Open"], data["High"], data["Low"], data["Close"]).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    dx = tb.DX(data["High"], data["Low"], data["Close"],
               timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    macdext, macdextsignal, macdexthist = tb.MACDEXT(
        data["Close"], fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
    macdext = macdext.to_frame().reset_index().set_axis(axes, axis=1)
    macdfix, macdfixsignal, macdfixhist = tb.MACDFIX(
        data["Close"], signalperiod=9)
    macdfix = macdfix.to_frame().reset_index().set_axis(axes, axis=1)
    mfi = tb.MFI(data["High"], data["Low"], data["Close"], data["Volume"],
                 timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    minus_di = tb.MINUS_DI(data["High"], data["Low"], data["Close"],
                           timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    minus_dm = tb.MINUS_DM(data["High"], data["Low"], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    mom = tb.MOM(data["Close"], timeperiod=10).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    plus_di = tb.PLUS_DI(data["High"], data["Low"], data["Close"],
                         timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    plus_dm = tb.PLUS_DM(data["High"], data["Low"], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    rocp = tb.ROCP(data["Close"], timeperiod=10).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    rocr = tb.ROCR(data["Close"], timeperiod=10).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    rocr100 = tb.ROCR100(data["Close"], timeperiod=10).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    slowk, slowd = tb.STOCH(data["High"], data["Low"], data["Close"], fastk_period=5,
                            slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    slowk = slowk.to_frame().reset_index().set_axis(axes, axis=1)
    slowd = slowd.to_frame().reset_index().set_axis(axes, axis=1)
    fastk, fastd = tb.STOCHF(
        data["High"], data["Low"], data["Close"], fastk_period=5, fastd_period=3, fastd_matype=0)
    fastk = fastk.to_frame().reset_index().set_axis(axes, axis=1)
    fastd = fastd.to_frame().reset_index().set_axis(axes, axis=1)
    fastkrsi, fastdrsi = tb.STOCHRSI(
        data["Close"], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    fastkrsi = fastkrsi.to_frame().reset_index().set_axis(axes, axis=1)
    fastdrsi = fastdrsi.to_frame().reset_index().set_axis(axes, axis=1)
    trix = tb.TRIX(data["Close"], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    ultosc = tb.ULTOSC(data["High"], data["Low"], data["Close"], timeperiod1=7,
                       timeperiod2=14, timeperiod3=28).to_frame().reset_index().set_axis(axes, axis=1)

    bbands_upperband, bbands_middleband, bbands_lowerband = tb.BBANDS(
        data['Close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    bbands_upperband = bbands_upperband.to_frame().reset_index().set_axis(axes, axis=1)
    bbands_middleband = bbands_middleband.to_frame().reset_index().set_axis(axes, axis=1)
    bbands_lowerband = bbands_lowerband.to_frame().reset_index().set_axis(axes, axis=1)
    dema = tb.DEMA(data['Close'], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    ht_trendline = tb.HT_TRENDLINE(
        data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    kama = tb.KAMA(data['Close'], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    ma = tb.MA(data['Close'], timeperiod=30, matype=0).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    midpoint = tb.MIDPOINT(data['Close'], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    midprice = tb.MIDPRICE(data["High"], data["Low"], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    sarext = tb.SAREXT(data["High"], data["Low"], startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0,
                       accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0).to_frame().reset_index().set_axis(axes, axis=1)
    trima = tb.TRIMA(data['Close'], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)

    ad = tb.AD(data["High"], data["Low"], data['Close'],
               data['Volume']).to_frame().reset_index().set_axis(axes, axis=1)
    adosc = tb.ADOSC(data["High"], data["Low"], data['Close'], data['Volume'],
                     fastperiod=3, slowperiod=10).to_frame().reset_index().set_axis(axes, axis=1)

    trange = tb.TRANGE(data["High"], data["Low"], data['Close']).to_frame(
    ).reset_index().set_axis(axes, axis=1)

    avgprice = tb.AVGPRICE(data['Open'], data["High"], data["Low"],
                           data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    medprice = tb.MEDPRICE(data["High"], data["Low"]).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    typprice = tb.TYPPRICE(data["High"], data["Low"], data['Close']).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    wclprice = tb.WCLPRICE(data["High"], data["Low"], data['Close']).to_frame(
    ).reset_index().set_axis(axes, axis=1)

    beta = tb.BETA(data["High"], data["Low"], timeperiod=5).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    correl = tb.CORREL(data["High"], data["Low"], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    linearreg = tb.LINEARREG(data['Close'], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    linearreg_angle = tb.LINEARREG_ANGLE(
        data['Close'], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    linearreg_intercept = tb.LINEARREG_INTERCEPT(
        data['Close'], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    linearreg_slope = tb.LINEARREG_SLOPE(
        data['Close'], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    tsf = tb.TSF(data['Close'], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    var = tb.VAR(data['Close'], timeperiod=5, nbdev=1).to_frame(
    ).reset_index().set_axis(axes, axis=1)

    '''
    PREPROCESSING INDICATOR DATA
    '''
    # List of (indicators) DataFrames, size=n_indicators
    indicators = [rsi, cmo, plus_di, minus_di, willr, cci, ultosc, aroonosc, mfi, mom, macd, macdfix, linearreg_angle, linearreg_slope, rocp, roc, rocr, rocr100, slowk, fastd, slowd, aroonup, aroondown, apo,
                  macdext, fastk, ppo, minus_dm, adosc, fastdrsi, fastkrsi, trange, trix, std, bop, var, plus_dm, correl, ad, beta, wclprice, typprice, avgprice, medprice, bbands_lowerband, linearreg, obv,
                  bbands_middleband, tema, bbands_upperband, dema, midprice, midpoint, wma, ema, ht_trendline, kama, sma, ma, adxr, adx, trima, linearreg_intercept, dx] # tsf deleted to get 64x64 image
    # 15x15 matrix of indicators
    # [rsi, cmo, willr, cci, macd, roc, ppo, std, tema, obv, wma, ema, sma, adx, sar]

    # Number of indicators (int)
    nIndicators = len(indicators)
    count = 0 
    indicators_str = [
    'rsi', 'cmo', 'plus_di', 'minus_di', 'willr', 'cci', 'ultosc', 'aroonosc', 'mfi', 'mom', 'macd', 'macdfix',
    'linearreg_angle', 'linearreg_slope', 'rocp', 'roc', 'rocr', 'rocr100', 'slowk', 'fastd', 'slowd', 'aroonup',
    'aroondown', 'apo', 'macdext', 'fastk', 'ppo', 'minus_dm', 'adosc', 'fastdrsi', 'fastkrsi', 'trange', 'trix',
    'std', 'bop', 'var', 'plus_dm', 'correl', 'ad', 'beta', 'wclprice', 'typprice', 'avgprice', 'medprice',
    'bbands_lowerband', 'linearreg', 'obv', 'bbands_middleband', 'tema', 'bbands_upperband', 'dema', 'midprice',
    'midpoint', 'wma', 'ema', 'ht_trendline', 'kama', 'sma', 'ma', 'adxr', 'adx', 'trima', 'linearreg_intercept', 'dx'
    ]
    zipped = zip(indicators, indicators_str)
    for ind, name in zipped:
        # print(f"df {name}: {ind['Value']}")
        count += ind['Value'].isnull().sum()
    print(f"count = {count}")
    # Calculating the most number of null values in an indicator DataFrame's "Value" column
    maxNullVal = -1
    for indicator in indicators:
        if(indicator['Value'].isnull().sum() > maxNullVal):
            maxNullVal = indicator['Value'].isnull().sum()

    # List of (indicators "Value" column) DataFrames, size=n_indicators
    indicatorValues = []
    for indicator in indicators:
        # Getting rid of null values
        indicatorValues.append(indicator['Value'].iloc[maxNullVal:])

    # DataFrame, size=(n_days, n_indicators, col_names=headers)
    indicatorValuesMatrix = pd.concat(indicatorValues, axis=1, keys=headers)
    indicatorCorr = indicatorValuesMatrix.corr(method='pearson')

    '''
    dictCor = {}
    for header, value in zip(headers, indicatorCorr.iloc[0]):
        dictCor[header] = value
    sortedDictCor = {k: v for k, v in sorted(dictCor.items(), key=lambda item: abs(item[1]), reverse=True)}
    for k,v in sortedDictCor.items():
        print(k, v)

    '''

    '''
    CREATING THE IMAGES
    '''

    indicator_dict = {name: df['Value'] for df, name in zip(indicators, indicators_str)}
    maxNullVal = max(df['Value'].isnull().sum() for df in indicators)
    for name in indicators_str:
        indicator_dict[name] = indicator_dict[name].iloc[maxNullVal:]
    
    # below is for making it more uncorrelated
    distance_matrix = 1 - abs(indicatorCorr)  # Convert correlation to distance

    pca = PCA(n_components=2)  # n_components can be adjusted
    reduced_data = pca.fit_transform(distance_matrix)

    # Apply K-means clustering on the reduced data
    kmeans = KMeans(n_clusters=10, random_state=0).fit(reduced_data)

    # Get cluster labels for each indicator
    clusters = kmeans.labels_
    

    from itertools import cycle, islice
    
    cluster_dict = {}
    for label, indicator in zip(clusters, indicators_str):
        cluster_dict.setdefault(label, []).append(indicator)

    unique_clusters = sorted(cluster_dict.keys())
    selected_indicators = set()
    ordered_indicators = []
    # round robin but bounded by 64
    for cluster in cycle(unique_clusters):
        if len(ordered_indicators) >= 64:
            break

        # Pick one indicator from the current cluster
        for indicator in cluster_dict[cluster]:
            if indicator not in selected_indicators:
                ordered_indicators.append(indicator)
                selected_indicators.add(indicator)
                break
    # print(f"len of ordered: {len(ordered_indicators)}")
    print(f"{sorted(ordered_indicators) == sorted(indicators_str)}") # so thats okay we matched
    nDays = len(indicator_dict[ordered_indicators[0]])
    for idx in range(nDays - 2 * nIndicators):
        image = []
        for indicator_name in ordered_indicators:
            # Retrieve the corresponding row (indicator values) for this indicator
            indicator_values = indicator_dict[indicator_name]
            imageRow = indicator_values.iloc[idx:idx + nIndicators].values
            image.append(imageRow)
        print(np.array(image).shape)
        imageList.append(np.array(image))

        
    
    '''
    CREATING THE LABELS
    '''
    print(f"mxnullvall: {maxNullVal}")
    # Pandas Series, size=n_days-(maxNullVal+nIndicators-1) -> Check this, size is imageList+1, might be a bug.
    data_close = data[maxNullVal+2*nIndicators-1:]["Close"]
    data_adj_close = data[maxNullVal+2*nIndicators-1:]["Adj Close"][:-1]
    print(f"data_close: {len(data_close)}")
    print(f"data_adj_close: {len(data_adj_close)}")
    
    # Buy : 0
    # Hold: 1
    # Sell: 2
    for i in range((len(data_close)-1)):
        closePriceDifference = data_close.iloc[i+1] - data_close.iloc[i]
        thresholdPrice = threshold * data_close.iloc[i]
        # If the price has increased
        if(closePriceDifference > 0):
            # but not enough to pass the threshold
            if(closePriceDifference <= thresholdPrice):
                labelList.append(np.array([1.0]))  # HOLD
            # enough to pass the threshold
            else:
                labelList.append(np.array([0.0]))  # BUY
        # If the price has decreased
        elif(closePriceDifference < 0):
            # but not so much to pass the thresshold
            if(abs(closePriceDifference) <= thresholdPrice):
                labelList.append(np.array([1.0]))  # HOLD
            # so much to pass the threshold
            else:
                labelList.append(np.array([2.0]))  # SELL
        # If the price hasn't changed
        else:
            labelList.append(np.array([1.0]))  # HOLD
        
    
    
    print(len(labelList))
    standartized_image_list = []
    for img in imageList:
        m = np.mean(img, axis=1, keepdims=True)
        s = np.std(img, axis=1, keepdims=True)
        standartized_image = (img - m ) / s
        standartized_image_list.append(standartized_image)
    
    x_train = []
    y_train = []

    x_test = []
    y_test = []

    train_date = []
    test_date = []

    train_price = []
    test_price = []
    print(f"label list: {len(labelList)}")
    print(f"std_img: {len(standartized_image_list)}")
    for index in range(len(standartized_image_list)):
        if(index < (len(standartized_image_list) * 0.8)):
            x_train.append(standartized_image_list[index])
            y_train.append(labelList[index])
            train_date.append(data_close.index[index])
            train_price.append(data_close.iloc[index])
        else:
            x_test.append(standartized_image_list[index])
            y_test.append(labelList[index])
            test_date.append(data_close.index[index])
            test_price.append(data_close.iloc[index])

    np.save(f"/home/emir/Desktop/dev/datasets/ETF/rectangle/01/TrainData/x_{etf}.npy", x_train)
    np.save(f"/home/emir/Desktop/dev/datasets/ETF/rectangle/01/TrainData/y_{etf}.npy", y_train)
    np.save(f"/home/emir/Desktop/dev/datasets/ETF/rectangle/01/TestData/x_{etf}.npy", x_test)
    np.save(f"/home/emir/Desktop/dev/datasets/ETF/rectangle/01/TestData/y_{etf}.npy", y_test)

    # np.save(f"/home/emir/Desktop/dev/datasets/ETF/rectangle/01/Date/TrainDate/{etf}.npy", train_date)
    # np.save(f"/home/emir/Desktop/dev/datasets/ETF/rectangle/01/Date/TestDate/{etf}.npy", test_date)
    # np.save(f'/home/emir/Desktop/dev/datasets/ETF/rectangle/01/Price/TrainPrice/{etf}.npy', train_price)
    # np.save(f'/home/emir/Desktop/dev/datasets/ETF/rectangle/01/Price/TestPrice/{etf}.npy', test_price)