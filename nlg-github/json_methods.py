import numpy as np
from scipy.stats import norm
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import signal
import datetime
from math import ceil
import math
from operator import itemgetter


'''
This is a file containing widget methods for analysis
'''

def dict_normalize(d, target=1.0):
    """ Normalize a dictionary.
    """
    raw = sum(d.values())
    factor = target / raw
    return {key: value * factor for key, value in d.items()}

def week_of_month(dt):
    """ Returns the week of the month for the specified date.
    """

    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()

    return int(ceil(adjusted_dom/7.0))


def mk_test(x, alpha=0.05):
    """
    This function is derived from code originally posted by Sat Kumar Tomer (satkumartomer@gmail.com)
    See also: http://vsp.pnnl.gov/help/Vsample/Design_Trend_Mann_Kendall.htm

    The purpose of the Mann-Kendall (MK) test (Mann 1945, Kendall 1975, Gilbert 1987) is to statistically assess if there is a monotonic upward or downward trend of the variable of interest over time. A monotonic upward (downward) trend means that the variable consistently increases (decreases) through time, but the trend may or may not be linear. The MK test can be used in place of a parametric linear regression analysis, which can be used to test if the slope of the estimated linear regression line is different from zero. The regression analysis requires that the residuals from the fitted regression line be normally distributed; an assumption not required by the MK test, that is, the MK test is a non-parametric (distribution-free) test.
    Hirsch, Slack and Smith (1982, page 107) indicate that the MK test is best viewed as an exploratory analysis and is most appropriately used to identify stations where changes are significant or of large magnitude and to quantify these findings.

    Input:
        x:   a vector of data
        alpha: significance level (0.05 default)

    Output:
        trend: tells the trend (increasing, decreasing or no trend)
        h: True (if trend is present) or False (if trend is absence)
        p: p value of the significance test
        z: normalized test statistics

    Examples
    --------
      >>> x = np.random.rand(100)
      >>> trend,h,p,z = mk_test(x,0.05)
    """
    n = len(x)

    # calculate S
    s = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            s += np.sign(x[j] - x[k])

    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)

    # calculate the var(s)
    if n == g:  # there is no tie
        var_s = (n * (n - 1) * (2 * n + 5)) / 18
    else:  # there are some ties in data
        tp = np.zeros(unique_x.shape)
        for i in range(len(unique_x)):
            tp[i] = sum(x == unique_x[i])
        var_s = (n * (n - 1) * (2 * n + 5) - np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s == 0:
        z = 0
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)

    # calculate the p_value
    p = 2 * (1 - norm.cdf(abs(z)))  # two tail test
    h = abs(z) > norm.ppf(1 - alpha / 2)

    if (z < 0) and h:
        trend = 'downward trend'
    elif (z > 0) and h:
        trend = 'upward trend'
    else:
        trend = 'neither obvious upward nor downward trend'

    return trend, h, p, z

def time_series_analysis(x, interval):
    """ input is an np array (series) and its unit name in string.
    """

    avglvl = np.mean(x)
    # periodicity analysis, which employ fourier transformation to detect possible
    # frequencies/periodicity in a series.
    # details can be checked at:
    # http://www.l3s.de/~anand/tir14/lectures/ws14-tir-foundations-2.pdf
    freqcandidate, Pxx_den = signal.periodogram(x)

    tgt_ind = np.argmax(Pxx_den)
    # Ignore the periodicity longer than the whole series.
    while round(1 / freqcandidate[np.argmax(Pxx_den)]) >= x.shape[0]:
        Pxx_den = np.delete(Pxx_den, tgt_ind)
        tgt_ind = np.argmax(Pxx_den)

    frq = int(1 / freqcandidate[np.argmax(Pxx_den)])

    # Use the found frequency to conduct seasonality decomposition
    decomposition = seasonal_decompose(x, freq=frq, model='additive')

    # you can visialize the decomposition result here:
    #decplot = decomposition.plot()
    #decplot.show()

    # Check the amplification of fluctuation.
    amplification = max(abs(decomposition.seasonal))
    apratio = amplification / avglvl
    oritrend = decomposition.trend[~np.isnan(decomposition.trend)]

    # Use mk_test the analyze the trend that has been de-periodized.
    trend, h, p, z = mk_test(oritrend, alpha=0.5)

    # Matching proper wordings to various results.

    if p < 0.1:
        trend = 'an overall significant ' + trend
    if 0.1 <= p < 0.5:
        trend = 'an overall plausible ' + trend

    if frq > 1:
        unit = ' ' + interval + 's'
    else:
        unit = ' ' + interval

    if apratio < 0.01 or math.isnan(apratio):
        periodicity = ''
    elif 0.01 <= apratio < 0.05:
        periodicity = 'slight periodicity of ' + str(frq) + unit
    elif 0.05 <= apratio < 0.15:
        periodicity = 'moderate periodicity of ' + str(frq) + unit
    elif 0.15 <= apratio :
        periodicity = 'evident periodicity of ' + str(frq) + unit

    return [trend, periodicity, apratio, frq, p]


def time_unit(start, end, interval):
    """ translate the time (micro-seconds to a nature language style.)
    """

    month_list = [0, 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    st = datetime.datetime.fromtimestamp(start / 1000).strftime('%Y, %m, %d, %I, %p, %M, %S, %f').split(',')

    ed = datetime.datetime.fromtimestamp(end / 1000).strftime('%Y, %m, %d, %I, %p, %M, %S, %f').split(',')

    minu = 60
    hr = 60
    day = 24
    week = 7
    month = 4
    season = 3
    year = 4


    time_unit = [minu, hr, day, week, month, season, year, 10]
    time_metric_list = ['second', 'minute', 'hour', 'day', 'week', 'month', 'season', 'year', 'decade']

    div = interval / 1000

    for i, elm in enumerate(time_unit):   # translate the time unit, possible unit names are in time_metric_list
        if int(div) <= 1:
            time_metric = time_metric_list[i]
            if i > 4: resolution = 'coarse'
            else: resolution = 'precise'
            break
        div, s = divmod(div, elm)

    # add proper suffix to date.
    stDAY = Date_judge(st)
    edDAY = Date_judge(ed)


    if resolution == 'precise':
        return str(month_list[int(st[1])]) + str(stDAY) + ',' + str(st[3]) + ':' + str(st[5][1:]) + str(st[4]) + ', ' + str(st[0]), \
               str(month_list[int(ed[1])]) + str(edDAY) + ',' + str(ed[3]) + ':' + str(ed[5][1:]) + str(ed[4]) + ', ' + str(ed[0]), \
               time_metric, st, ed
    else:
        return str(month_list[int(st[1])]) + str(stDAY) + ', ' + str(st[0]), \
               str(month_list[int(ed[1])]) + str(edDAY) + ', ' + str(ed[0]), \
               time_metric, st, ed


def Date_judge(st):
    for i in [2, 3, 5, 6]:
        if st[2][1] == '0':
            st[2] = ' ' + st[2][2:]

    if list(st[2])[-1] == '1':
        DAY = st[2] + 'st'

    elif list(st[2])[-1] == '2':
        DAY = st[2] + 'nd'
    elif list(st[2])[-1] == '3':
        DAY = st[2] + 'rd'
    else:
        DAY = st[2] + 'th'

    return DAY

def Date_judge_str(st):
    if list(st)[1] == '0':
        st = ' ' + ''.join(list(st)[2:])

    if list(st)[-1] == '1':
        DAY = st + 'st'
    elif list(st)[-1] == '2':
        DAY = st + 'nd'
    elif list(st)[-1] == '3':
        DAY = st + 'rd'
    else:
        DAY = st + 'th'

    return DAY

def Time_judge_str(st):
    if list(st)[1] == '0':
        st = ' ' + ''.join(list(st)[2:])



    return st


def Time_Slot_Trans(num, st, ed, interval, intdate):
    """ translate the time unit in a nature language style.
    """


    Time_Slot_dt = datetime.datetime.fromtimestamp(st / 1000) + datetime.timedelta(seconds = num * interval / 1000)
    Time_Slot = Time_Slot_dt.strftime('%Y, %m, %d, %I, %p, %M, %S, %f').split(',')

    month_list = [0, 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    time_metric_list = ['second', 'minute', 'hour', 'day', 'week', 'month', 'season', 'year']

    if intdate == 'season':
        month = int(Time_Slot[1])
        sea, mon = divmod(month, 3)
        if mon > 0:
            sea += 1
        rst_Time_Slot = 'Q'+str(sea) + ' ' + Time_Slot[0]

    elif intdate == 'year':

        rst_Time_Slot = Time_Slot[0] + ' year'


    elif intdate == 'week':
        week_num = week_of_month(Time_Slot_dt)

        rst_Time_Slot = 'the ' + Date_judge_str(str(week_num)) + ' week of ' + time_metric_list[Time_Slot[1]] + ', ' + str(Time_Slot[0])

    elif intdate == 'month':

        rst_Time_Slot = time_metric_list[Time_Slot[1]] + ', ' + str(Time_Slot[0])


    elif intdate == 'day':

        rst_Time_Slot = time_metric_list[Time_Slot[1]] + ' ' + str(Date_judge_str(Time_Slot[2])) + ', ' + str(Time_Slot[0])


    elif intdate == 'hour':

        rst_Time_Slot = Time_Slot[3] + ' ' + Time_Slot[5] + ', ' + time_metric_list[Time_Slot[1]] + ' ' + str(Date_judge_str(Time_Slot[2])) + ', ' + str(Time_Slot[0])

    elif intdate == 'minute':

        rst_Time_Slot = month_list[int(Time_Slot[1])] + str(Date_judge_str(Time_Slot[2])) + ',' + Time_judge_str(str(Time_Slot[3])) + ':' +str(Time_Slot[5]) + '' + Time_Slot[4] + ', ' + str(Time_Slot[0])


    elif intdate == 'second':

        rst_Time_Slot = Time_Slot[3] + ' : ' + Time_Slot[4] + ' : ' + Time_Slot[6] + ' ' + Time_Slot[5] + ', ' + time_metric_list[Time_Slot[1]] + ' ' + str(Date_judge_str(Time_Slot[2])) + ', ' + str(Time_Slot[0])


    return rst_Time_Slot



def top_cat_words(l, prop, words_display_num):
    """ Retrieve the top words_display_num words associated with each topic.
    """

    count_topics = {}
    for i in range(prop.shape[0] - 1, -1, -1):
        curbw = l[int(prop[i][1])]['buzzwords']
        tt_count = l[int(prop[i][1])]['c']
        curbw = sorted(curbw, key=itemgetter('c'), reverse=True)
        sub_count_topics = []
        for j in range(words_display_num):
            ct = curbw[j]['c']
            word = curbw[j]['term'].replace('_', ' ')
            sub_count_topics.append([ct, word, tt_count])
        count_topics[str(int(prop[i][1]))] = sub_count_topics
    return count_topics


def mad_based_outlier(points, thresh=3.5):
    """ Outlier detection algorithm (MAD).
    """

    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh