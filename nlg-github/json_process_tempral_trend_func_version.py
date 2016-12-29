import numpy as np
import json
import pandas as pd
import operator
from json_methods import Time_Slot_Trans, dict_normalize, time_series_analysis, time_unit

""" The temporal trend is pretty complicated. So I use a single script to implement it.
"""

def temp_trend(tpc, slt, start, end, interval, l):
    """ The inputs are:
        tpc: topic number
        slt: slot length
        start: start time
        end: end time
        interval: time unit
        l: the temporal trend data
    """


    words_display_num = 5
    stdate, eddate, intdate, st, ed = time_unit(start, end, interval) # Translate the time in nature language style.


    mymatrix = np.zeros((tpc, slt)) # Storing count info
    mysentPmatrix = np.zeros((tpc, slt))  # Storing positive info
    mysentNmatrix = np.zeros((tpc, slt)) # Storing negative info
    mykwdmatrix = []
    keywordmatrix = []
    status = {}
    total_key_word_dict = {}
    for i, elm in enumerate(l['bins']):
        mymatrix[i][:] = np.array(elm['c'])

        mysentPmatrix[i][:] = np.array(elm['p'])
        mysentNmatrix[i][:] = np.array(elm['n'])

        x = np.array(elm['c'])
        x = x[np.nonzero(x)]
        status[i] = time_series_analysis(x, intdate)


        '''
        find key words with respect to counts
        '''

        curslot = list(filter(None, elm['terms']))
        key_word_dict = {}
        for key_words in curslot:
            for key_word in key_words:
                if key_word['text'] in key_word_dict.keys():
                    key_word_dict[key_word['text']] += key_word['c']
                else:
                    key_word_dict[key_word['text']] = key_word['c']

                if key_word['text'] in total_key_word_dict.keys():
                    total_key_word_dict[key_word['text']] += key_word['c']
                else:
                    total_key_word_dict[key_word['text']] = key_word['c']
        mykwdmatrix.append(dict_normalize(key_word_dict))

        # Sort the words by total count
        sorted_key_word = sorted(key_word_dict.items(), key=operator.itemgetter(1), reverse=True)
        test = [elm[0] for elm in sorted_key_word[0:words_display_num]]
        keywordmatrix.append(test)

    # Sort words for each topic and store the top words_display_num words
    finalmykwdmatrix = []
    for ind, sbtpc in enumerate(range(tpc)):
        curkey_word_dict = mykwdmatrix[ind]
        sorted_key_word = sorted(curkey_word_dict.items(), key=operator.itemgetter(1), reverse=True)
        test = [elm for elm in sorted_key_word[0:words_display_num]]
        finalmykwdmatrix.append(test)


    total_sorted_key_word = sorted(total_key_word_dict.items(), key=operator.itemgetter(1), reverse=True)
    total_test = [elm for elm in total_sorted_key_word[0:words_display_num]]
    total_amount = sum([elm[1] for elm in total_sorted_key_word[0:words_display_num]])

    '''
    time trend analysis for the total counts
    '''

    sumx = np.sum(mymatrix, axis=0)
    sumx = sumx[np.nonzero(sumx)]
    sum_status = time_series_analysis(sumx, intdate)

    # find the peak and valley time and translate them into nature language style.
    peak_time = Time_Slot_Trans(np.argmax(sumx) + 1, start, end, interval, intdate)
    valley_time = Time_Slot_Trans(np.argmin(sumx) + 1, start, end, interval, intdate)


    if not sum_status[1]: # if all of the topics do not have periodicity.
        print('For all of the documents, the temporal pattern exhibits ' + sum_status[0] + ' over the period from ' + stdate + ' to '+ eddate + '.')
        print('It reaches peak at ' + peak_time + ' and touches valley at ' + valley_time + '.')
    else: # else output propoer wordings for various situations.
        if sum_status[0].find('significant'):
            if sum_status[1]:
                print('For all of the documents, the temporal pattern exhibits ' + sum_status[0] + ' with ' + sum_status[1] + ' over the period from ' + stdate + ' to '+ eddate + '.')

                print('It reaches peak at ' + peak_time + ' and touches valley at ' + valley_time + '.')

        elif sum_status[0].find('plausible'):
            if sum_status[1].find('slight'):
                print('For all of the documents, the temporal pattern exhibits ' + sum_status[0] + ' with ' + sum_status[1] + ' over the period from ' + stdate + ' to '+ eddate + '.')
                print('It reaches peak at ' + peak_time + ' and touches valley at ' + valley_time + '.')
            elif sum_status[1]:
                print('For all of th documents, the temporal pattern exhibits ' + sum_status[1] + ' with ' + sum_status[0] + ' over the period from ' + stdate + ' to '+ eddate + '.')
                print('It reaches peak at ' + peak_time + ' and touches valley at ' + valley_time + '.')

    '''
    The count correlation over time for each topic, comparing to the total trend
    '''
    print(' ')

    tplist = [i for i in range(mymatrix.shape[0])]
    ts0 = pd.Series(sumx)
    pearsonr = []

    # calculate pearson correlation for each topic to the total trend.
    for elm in tplist:
        ts1 = pd.Series(mymatrix[elm][np.nonzero(mymatrix[elm][:])])
        pearsonr.append(ts0.corr(ts1))

    # proper wordings for various correlation results.
    corrwords = ['strongly related',
                 'weakly related',
                 'not related',
                 'weakly contrary',
                 'strongly contrary'
                 ]

    pearsonr = np.asarray(pearsonr)
    rsts = [pearsonr[np.where(pearsonr >= 0.7)],
            pearsonr[(pearsonr >= 0.3) & (pearsonr < 0.7)],
            pearsonr[(pearsonr >= -0.3) & (pearsonr < 0.3)],
            pearsonr[(pearsonr >= -0.7) & (pearsonr < -0.3)],
            pearsonr[(pearsonr >= -1) & (pearsonr < -0.3)]
            ]

    for i, rst in enumerate(rsts):
        if rst.shape[0] > 0:
            if rst.shape[0] == pearsonr.shape[0]:
                print('All of the categories in the documents are ' + str(corrwords[i]) + ' with the change of the total trend.')
            else:
                if rst.shape[0] == 1:
                    category = ' categories is '
                else:
                    category = ' categories are '
                print(str(rst.shape[0]) + ' of the total ' + str(pearsonr.shape[0]) + category + str(
                    corrwords[i]) + ' with the total document amount trend.')


    '''
    key words summary
    '''

    if len(total_test) > 1:
        totalwordList = ", ".join(str(e) for e in total_test[:-1])
        totalwordList = totalwordList + ' and ' + str(total_test[-1]) + ', etc.'
        print('Among these categories, the key words that generate most of the discussions are:')
        for e in total_test:
            print(str(e[0]) + ' (' + '{:.1%}'.format(e[1]/total_amount) + ' of the documents)')
    else:
        totalwordList = total_test[0]
        print('Among these categories, ' + str(totalwordList) + ' generates most of the discussions.' + ' ( ' + '{:.1%}'.format(total_test[1]/total_amount) + ' of the documents )')



    '''
    Sentiment Summary. The logic is similar as count, but the interested variables are changed to sentiment related.
    '''
    print(' ')
    mysentmatrix = np.add(mysentNmatrix, mysentPmatrix)
    mysentmatrix = np.divide(mysentmatrix, mymatrix) # normalize sentiment score with respect to count.
    sumsentx = np.sum(mysentmatrix, axis=0)
    sumsentx[np.argwhere(np.isnan(sumsentx))] = 0
    sumsentx_ori = sumsentx[np.nonzero(sumsentx)]
    sumsentx = np.divide(sumsentx_ori, sumx)

    # Find peak and valley sentiment time and translate it into nature languange style.
    sent_peak_time = Time_Slot_Trans(np.argmax(sumsentx) + 1, start, end, interval, intdate)
    sent_valley_time = Time_Slot_Trans(np.argmin(sumsentx) + 1, start, end, interval, intdate)

    sumsent_status = time_series_analysis(sumsentx, intdate)
    avgsent = np.mean(sumsentx)

    if avgsent > 0: #if the overall sentiment is positive.

        opt = 'The overall average sentiment for all of the documents over the period from ' + stdate + ' to '+ eddate + ' is positive. '

        if not sumsent_status[1]:   #if there is periodicty

            if sumsent_status[0].find('significant upward'):
                print(
                    opt + 'For the total number of positive sentiment documents, the temporal pattern exhibits ' + sumsent_status[0] + ' with ' + sumsent_status[1] + ' during the period.')

                print('It reaches peak at ' + sent_peak_time + ' and touches valley at ' + sent_valley_time + '.')

            elif sumsent_status[0].find('significant downward'):

                opt = 'Although the overall average sentiment for all of the documents over the period from ' + stdate + ' to '+ eddate + ' is positive, '

                print(opt + 'for the total number of positive sentiment documents, the temporal pattern exhibits ' + sumsent_status[0] + ', with ' + sumsent_status[1] + ' during this period.')

                print('It touches valley at ' + sent_valley_time + ' and reaches peak at ' + sent_peak_time + '.')

            elif sumsent_status[0].find('plausible upward'):

                if sumsent_status[1].find('slight'):
                    print(opt + 'For the total number of positive sentiment documents, the temporal pattern exhibits ' + sumsent_status[0] + ', with ' + sumsent_status[1] + ' during the period.')
                    print('It reaches peak at ' + sent_peak_time + ' and touches valley at ' + sent_valley_time + '.')
                else:
                    print(opt + 'For the total number of positive sentiment documents, the temporal pattern exhibits ' + sumsent_status[1] + ', with ' + sumsent_status[0] + ' during the period.')
                    print('It reaches peak at ' + sent_peak_time + ' and touches valley at ' + sent_valley_time + '.')


            elif sumsent_status[0].find('plausible downward'):

                opt = 'Although the overall average sentiment for all of the documents over the period from ' + stdate + ' to '+ eddate + ' is positive, '

                if sumsent_status[1].find('slight'):
                    print(
                        opt + 'for the total number of positive sentiment documents, the temporal pattern exhibits ' + sumsent_status[0] + ', with ' + sumsent_status[1] + ' during the period.')

                    print('It touches valley at ' + sent_valley_time + ' and reaches peak at ' + sent_peak_time + '.')
                else:
                    print(
                        opt + 'for the total number of positive sentiment documents, the temporal pattern exhibits ' + sumsent_status[1] + ', with ' + sumsent_status[0] + ' during the period.')
                    print('It touches valley at ' + sent_valley_time + ' and reaches peak at ' + sent_peak_time + '.')

        else: #if there is no periodicty
            if sumsent_status[0].find('significant upward'):
                print(opt + 'For the total number of positive sentiment documents, the temporal pattern exhibits ' + sumsent_status[0] + ' during the period.')

                print('It reaches peak at ' + sent_peak_time + ' and touches valley at ' + sent_valley_time + '.')

            elif sumsent_status[0].find('significant downward'):

                opt = 'Although the overall average sentiment for all of the documents over the period from ' + stdate + ' to '+ eddate + ' is positive, '


                print(opt + 'for the total number of positive sentiment documents, the temporal pattern exhibits ' + sumsent_status[0] + ' during the period.')

                print('It touches valley at ' + sent_valley_time + ' and reaches peak at ' + sent_peak_time + '.')

            elif sumsent_status[0].find('plausible upward'):

                if sumsent_status[1].find('slight'):
                    print(opt + 'And for the total number of positive sentiment documents, the temporal pattern exhibits ' + sumsent_status[0] + ' during the period.')
                    print('It reaches peak at ' + sent_peak_time + ' and touches valley at ' + sent_valley_time + '.')
                else:
                    print(opt + 'And for the total number of positive sentiment documents, the temporal pattern exhibits ' + sumsent_status[0] + ' during the period.')
                    print('It reaches peak at ' + sent_peak_time + ' and touches valley at ' + sent_valley_time + '.')


            elif sumsent_status[0].find('plausible downward'):

                opt = 'Although the overall average sentiment for all of the documents over the period from ' + stdate + ' to '+ eddate + ' is positive, '

                if sumsent_status[1].find('slight'):
                    print(opt + 'and for the total number of positive sentiment documents, the temporal pattern exhibits ' + sumsent_status[0] + ' during the period.')

                    print('It touches valley at ' + sent_valley_time + ' and reaches peak at ' + sent_peak_time + '.')
                else:
                    print(
                        opt + 'and for the total number of positive sentiment documents, the temporal pattern exhibits ' + sumsent_status[0] + ' during the period.')
                    print('It touches valley at ' + sent_valley_time + ' and reaches peak at ' + sent_peak_time + '.')

    else:
        opt = 'The overall average sentiment for all of the documents over the period from ' + stdate + ' to '+ eddate + ' is negative. '

        if not sumsent_status[1]:
            print(opt + 'For the total negative sentiment documents, the temporal pattern exhibits ' + sumsent_status[0] + ' during the period.')
        else:

            if sumsent_status[0].find('significant upward'):

                opt = 'Although the overall average sentiment for all of the documents over the period from ' + stdate + ' to '+ eddate + ' is negative, '

                print(
                    opt + 'there is a significant improvement for the total sentiment scores, associated with ' +
                    sumsent_status[
                        1] + ' during the period.')

                print('It reaches peak at ' + sent_peak_time + ' and touches valley at ' + sent_valley_time + '.')

            elif sumsent_status[0].find('significantly decreasing'):

                #opt = 'Although the overall average sentiment over the period from xxx to xxx is positive, '

                print(opt + 'What is worse, there is a sever deterioration for the total sentiment scores, associated with ' +
                      sumsent_status[
                          1] + ' during this period.')

                print('It touches valley at ' + sent_valley_time + ' and reaches peak at ' + sent_peak_time + '.')

            elif sumsent_status[0].find('plausibly increasing'):

                opt = 'Although the overall average sentiment over the period from ' + stdate + ' to '+ eddate + ' is negative, '


                if sumsent_status[1].find('slight'):

                    print(opt + 'there is ' + sumsent_status[
                        0] + ' for the total trend, associated with slightly improvement during this period.')
                    print('It reaches peak at ' + sent_peak_time + ' and touches valley at ' + sent_valley_time + '.')
                else:
                    print(opt + 'there is ' + sumsent_status[
                        1] + ' for the total trend, associated with slightly improvement during this period.')
                    print('It reaches peak at ' + sent_peak_time + ' and touches valley at ' + sent_valley_time + '.')


            elif sumsent_status[0].find('plausibly decreasing'):

                if sumsent_status[1].find('slight'):
                    print(
                        opt + 'What is worse, there is a plausible deterioration for the total sentiment scores, associated with ' +
                        sumsent_status[
                            1] + ' during this period.')

                    print('It touches valley at ' + sent_valley_time + ' and reaches peak at ' + sent_peak_time + '.')
                else:
                    print(
                        opt + 'What is worse, there is ' + sumsent_status[1] + ' for the total sentiment scores, associated with a plausible deterioration during this period.')
                    print('It touches valley at ' + sent_valley_time + ' and reaches peak at ' + sent_peak_time + '.')


    '''
        The time correlation for sub-cat with total trend
    '''

    tplist = [i for i in range(mysentmatrix.shape[0])]
    ts0 = pd.Series(sumsentx)

    pearsonr = []
    for i, elm in enumerate(tplist):
        ts1 = pd.Series(np.divide(mysentmatrix[elm][np.nonzero(mymatrix[elm][:])], mymatrix[elm][np.nonzero(mymatrix[elm][:])]))
        pearsonr.append([ts0.corr(ts1)])



    corrwords = ['strongly related',
                 'weakly related',
                 'not related',
                 'weakly contrary',
                 'strongly contrary'
                 ]

    mykwdmatrix = np.array(finalmykwdmatrix)
    pearsonr = np.asarray(pearsonr)
    rsts = [pearsonr[np.where(pearsonr >= 0.7)],
            pearsonr[(pearsonr >= 0.3 ) & (pearsonr < 0.7)],
            pearsonr[(pearsonr >= -0.3) & (pearsonr < 0.3)],
            pearsonr[(pearsonr >= -0.7 ) & (pearsonr < -0.3)],
            pearsonr[(pearsonr >= -1) & (pearsonr < -0.7)]
           ]

    er = np.where(pearsonr >= 0.7)
    wr = np.where(np.logical_and(pearsonr >= 0.3, pearsonr < 0.7))
    nr = np.where(np.logical_and(pearsonr >= -0.3, pearsonr < 0.3))
    wc = np.where(np.logical_and(pearsonr >= -0.7, pearsonr < - 0.3))
    ec = np.where(np.logical_and(pearsonr >= -1, pearsonr < - 0.7))
    tpcID = [er[0], wr[0], nr[0], wc[0], ec[0]]


    tpcrsts = [mykwdmatrix[np.where(pearsonr >= 0.7)[0]][:][:],
               mykwdmatrix[wr[0]][:][:],
               mykwdmatrix[nr[0]][:][:],
               mykwdmatrix[wc[0]][:][:],
               mykwdmatrix[ec[0]][:][:]
            ]


    for i, rst in reversed(list(enumerate(rsts))):
        if rst.shape[0] > 0:
            if rst.shape[0] == pearsonr.shape[0]:
                if i < 3:
                    print('All of topics are ' + str(corrwords[i]) + ' in sentiment to the overall trend.')
                else:
                    print('All of topics are ' + str(
                        corrwords[i]) + ' in sentiment to the overall trend.')
            else:
                if rst.shape[0] == 1:
                    tpcs = ' topics is '
                    be = 'The topic id and frequently mentioned words in this topic include: '
                elif rst.shape[0] > 1:
                    tpcs = ' topics are '
                    be = 'The topic ids and frequently mentioned words in these topics include: '
                print(str(rst.shape[0]) + ' of the ' + str(pearsonr.shape[0]) + tpcs + str(corrwords[i]) + ' in sentiment to the overall trend.')
                if i > 3:
                    print(be)
                    for tpcid, elm in enumerate(tpcrsts[i]):

                        keywordList = ", ".join(e[0] + ' (' + '{:.1%}'.format(float(e[1])) + ')' for e in tpcrsts[i][tpcid])
                        keywordList = keywordList + ', etc.'
                        print('No.' + str(tpcID[i][tpcid]) + ' topic: ' + keywordList)



def json_read(json_file_name):
    with open(json_file_name, 'r') as f:
        for line in f:
            yield json.loads(line)
