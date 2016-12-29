import numpy as np
import json
from operator import itemgetter
from json_methods import time_unit, top_cat_words, mad_based_outlier
from json_process_tempral_trend_func_version import temp_trend



'''
This code is used for analyzing the data in a semantic topic figure, which consists of 4 parts: category, buzzword,
temporal trend and geographical distribution.
The input of the code is a json file containing the data and the output of the code is a paragraph of words.
As an example, the path of the json file analyzed here is: data/categories.json
The path of the corresponding figure is: data/Semantic.png
'''



json_file_name = 'data/categories.json'   # the input json file

words_display_num = 5 # display word number
topics_display_num = 5 # display topic number


def json_read(json_file_name):
    with open(json_file_name, 'r') as f:
        for line in f:
            yield json.loads(line)

count = 0 #document count
ct_p = 0 #positive count
ct_n = 0 #negative count


for l in json_read(json_file_name):
    slt = len(l[0]['timebin']['bins'][0]['c'])  # temporal trend info
    start = l[0]['timebin']['bucket_start'] # start time
    end = l[0]['timebin']['bucket_end'] # end time
    interval = l[0]['timebin']['interval'] # interval

    stdate, eddate, intdate, st, ed = time_unit(start, end, interval) # transfer to readable time.
    tpcnum = len(l) # topic number
    count = np.zeros((tpcnum, 2))  # for document count info storage.
    sent = np.zeros((tpcnum, 2)) # for document sent info storage.

    region_info = {}  # for info storage of geo figure
    bzw_dict = {} # for info storage of buzzword figure
    tpc = []

    for i, elm in enumerate(l):
        # store the count and sentiment info for temporal trend
        tpc.append(elm['timebin']['bins'][i])
        count[i][0] = elm['c']
        count[i][1] = i

        sent[i][0] = (elm['n'] + elm['p'])/elm['c']
        sent[i][1] = i

        # store the count and sentiment info for geo

        for region in elm['top_regions']:
            if region['display_name'] in region_info.keys():
                region_info[region['display_name']][0] += region['c']
                region_info[region['display_name']][1] += (region['n'] + region['p'])
            else:
                region_info[region['display_name']] = [region['c'], (region['n'] + region['p'])]

        # store the count and sentiment info for buzzword

        for bzw in elm['buzzwords']:

            if bzw['term'] in bzw_dict.keys():
                bzw_dict[bzw['term']][0] += bzw['c']*float(elm['weight'])  # account for the weight
                bzw_dict[bzw['term']][1] += bzw['n'] + bzw['p']
                bzw_dict[bzw['term']][2] += bzw['c']
            else:
                bzw_dict[bzw['term']] = [bzw['c']*float(elm['weight']), (bzw['n'] + bzw['p']), bzw['c']]


    # prepare for the temporal trend analysis in the end.
    dict_tpc = {'bins': tpc}


    # print out xxx document collected from xxx date to xxx date are categorized into xxx topics
    tpcount = 'The ' + str(int(np.sum(count[:, 0]))) + ' documents collected from ' + str(stdate) + ' to ' + \
              str(eddate) + ' are categorized into ' + str(tpcnum) + ' topics.'
    print(tpcount)

    """
    Buzzword analysis
    """

    # sort the buzz word for their count and sentiment (average by count)

    bzw_list = [[k, v[0], v[1] / v[2], v[2]] for k, v in bzw_dict.items()]
    buzzword_count = sorted(bzw_list, key=itemgetter(1), reverse=True)
    buzzword_sent = sorted(bzw_list, key=itemgetter(2), reverse=True)

    print('')

    buzzword_output = 'Top buzzwords and the sentiment (on a -5 ~ 5 scale) associated with them mentioned in these ' \
                      'documents include: '
    print(buzzword_output)

    for i in range(words_display_num):
        buzzword_op = str(buzzword_count[i][0]) + ': appeared in ' + str(buzzword_count[i][3]) + ' out of ' + str(int(np.sum(count[:, 0]))) + ' documents' \
                      + ' (' + '{:.2%}'.format(buzzword_count[i][3] / np.sum(count[:, 0])) + '), with overall ' + '{:.2f}'.format(buzzword_count[i][2]) + ' sentiment.'
        print(buzzword_op)

    print('')


    """
    Semantic topic analysis
    """

    count[:, 0] = count[:,0]/np.sum(count[:, 0])
    prop = count[count[:, 0].argsort()]
    sent = sent[sent[:, 0].argsort()]

    # retrieve the topics with key words + count/sent
    count_topics = top_cat_words(l, prop, words_display_num)
    sent_topics = top_cat_words(l, sent, words_display_num)


    propout = 'The proportion of each topic in the total documents ranges from ' + '{:.1%}'.format(prop[0][0]) + ' to ' + '{:.1%}'.format(prop[-1][0]) + '. '

    # Analyze if there is outlier in count among the topics.
    # For count, we are only interested in massive ones.
    prop_outliers = mad_based_outlier(prop[:, 0])
    tgtprop = prop[prop_outliers, :]
    ind = np.where(tgtprop[:, 0] > np.mean(prop[:, 0]))
    tgtprop = tgtprop[ind[0], :]


    if tgtprop.shape[0] == 0:
        top_count = 'The proportions are evenly distributed, with mean of ' + '{:.1%}'.format(np.mean(prop[:, 0])) + ' and SD of ' + '{:.2%}'.format(np.std(prop[:, 0])) + '.'
        print(propout + top_count)
    elif tgtprop.shape[0] == 1: # if there is only 1 outlier.
        top_count = 'There is one topic that captures comparably more documents: '
        print(propout)
        print(top_count)
        for i in range(tgtprop.shape[0]):
            tpc = 'Topic ' + str(int(tgtprop[i][1])) + ' (' + '{:.1%}'.format(
                tgtprop[tgtprop.shape[0] - 1 - i][0]) + '), ' + 'top keywords: '
            for j in range(words_display_num):
                if j < words_display_num - 1:
                    tpc = tpc + count_topics[str(int(tgtprop[i][1]))][j][1] + ', '
                else:
                    tpc = tpc + 'and ' + count_topics[str(int(tgtprop[i][1]))][j][1] + '.'

            print(tpc)
    elif tgtprop.shape[0] > 1: # If there are more than 1 outlier.
        top_count = 'The topics that capture comparably more documents include: '
        print(propout)
        print(top_count)
        for i in range(tgtprop.shape[0]):
            tpc = 'Topic ' + str(int(tgtprop[i][1])) + ' (' + '{:.1%}'.format(
                tgtprop[tgtprop.shape[0] - 1 - i][0]) + '), ' + 'top keywords: '
            for j in range(words_display_num):
                if j < words_display_num - 1:
                    tpc = tpc + count_topics[str(int(tgtprop[i][1]))][j][1] + ', '
                else:
                    tpc = tpc + 'and ' + count_topics[str(int(tgtprop[i][1]))][j][1] + '.'


            print(tpc)

    print('')

    sentout = 'The sentiment score of each topics ranges from ' + '{:.2f}'.format(sent[0][0]) + ' to ' + '{:.2f}'.format(sent[-1][0]) + ', out of a (-5, +5) scale. '

    # Analyze if there is outlier in sentiment among the topics.
    # For sentiment, we are interested in both the extremely negative and positive ones.
    sent_outliers = mad_based_outlier(sent[:, 0])
    tgtsent = sent[sent_outliers, :]
    ind_pos = np.where(tgtsent[:, 0] > np.mean(sent[:, 0])) # extremely positive
    tgtsent_pos = tgtsent[ind_pos[0], :]
    ind_neg = np.where(tgtsent[:, 0] < np.mean(sent[:, 0])) # extremely negative
    tgtsent_neg = tgtsent[ind_neg[0], :]



    if tgtsent_pos.shape[0] == 0 and tgtsent_neg.shape[0] == 0:
        top_sent = 'The sentiment scores are evenly distributed among all of the topics, with mean of ' + '{:.2f}'.format(np.mean(sent[:, 0])) + ' and SD of ' + '{:.2f}'.format(np.std(sent[:, 0])) + '.'
        print(sentout + top_sent)
    else:
        if tgtsent_pos.shape[0] > 0:
            if tgtsent_pos.shape[0] == 1:
                top_sent = 'There is one topic that captures comparably more negative sentiment: '
                print(top_sent)
                for i in range(tgtsent_pos.shape[0]):
                    tpc = 'Topic ' + str(sent_topics[str(int(tgtsent_pos[i][1]))][0][3]) + ' (' + '{:.2f}'.format(
                        tgtsent_pos[tgtsent_pos.shape[0] - 1 - i][0]) + '),' + 'top key words: '
                    for j in range(words_display_num):
                        if j < words_display_num - 1:
                            tpc = tpc + sent_topics[i][j][1] + ', '
                        else:
                            tpc = tpc + 'and ' + sent_topics[str(int(tgtsent_pos[i][1]))][j][
                                1] + '.'
                    print(tpc)

            elif tgtsent_pos.shape[0] > 1:
                top_sent = 'The topics that capture comparably more positive sentiment include: '
                print(top_sent)
                for i in range(tgtsent_pos.shape[0]):
                    tpc = 'Topic ' + str(sent_topics[str(int(tgtsent_pos[i][1]))][0][3]) + ' (' + '{:.2f}'.format(
                        tgtsent_pos[tgtsent_pos.shape[0] - 1 - i][0]) + '),' + 'top key words: '
                    for j in range(words_display_num):
                        if j < words_display_num - 1:
                            tpc = tpc + sent_topics[i][j][1] + ', '
                        else:
                            tpc = tpc + 'and ' + sent_topics[str(int(tgtsent_pos[i][1]))][j][
                                1] + '.'
                    print(tpc)

        if tgtsent_neg.shape[0] > 0:
            if tgtsent_neg.shape[0] == 1:
                top_sent = 'There is one topic that captures comparably more negative sentiment: '
                print(top_sent)
                for i in range(tgtsent_neg.shape[0]):
                    tpc = 'Topic ' + str(
                        sent_topics[str(int(tgtsent_neg[i][1]))][0][3]) + ' (' + '{:.2f}'.format(
                        tgtsent_neg[tgtsent_neg.shape[0] - 1 - i][0]) + '),' + 'top key words: '
                    for j in range(words_display_num):
                        if j < words_display_num - 1:
                            tpc = tpc + sent_topics[i][j][1] + ', '
                        else:
                            tpc = tpc + 'and ' + sent_topics[str(int(tgtsent_neg[i][1]))][j][
                                1] + '.'
                    print(tpc)

            elif tgtsent_neg.shape[0] > 1:
                top_sent = 'The topics that capture comparably more negative sentiment include: '
                print(top_sent)
                for i in range(tgtsent_neg.shape[0]):
                    tpc = 'Topic ' + str(
                        sent_topics[str(int(tgtsent_neg[i][1]))][0][3]) + ' (' + '{:.2f}'.format(
                        tgtsent_neg[tgtsent_neg.shape[0] - 1 - i][0]) + '),' + 'top key words: '
                    for j in range(words_display_num):
                        if j < words_display_num - 1:
                            tpc = tpc + sent_topics[i][j][1] + ', '

                        else:
                            tpc = tpc + 'and ' + sent_topics[str(int(tgtsent_neg[i][1]))][j][
                                1] + '.'

                    print(tpc)


    """
    Geo analysis
    """

    region_info_list = []
    unknown = False

    # retreive the geo info, label the unknown locations.

    for key, value in region_info.items():
        value[1] = value[1] / value[0]
        region_info_list.append([key, value[0], value[1]])
        if key == 'Unknown':
            unknown = True

    # Sort the geo info by count/sentiment
    region_info_count = sorted(region_info_list, key=itemgetter(1), reverse=True)
    region_info_sent = sorted(region_info_list, key=itemgetter(2), reverse=True)

    # Normalize the count and get the proportion.
    info_count = np.array([[elm[1], elm[2], i] for i, elm in enumerate(region_info_count)])
    total_count = np.sum(info_count[:, 0])
    info_count[:,0] = info_count[:,0] / np.sum(info_count[:,0])
    info_sent = np.array([[elm[1], elm[2], i] for i, elm in enumerate(region_info_sent)])

    # Conclude that these documents come from xxx countries.
    if unknown: # If there is unknown country, pick it out and describe it with 'other countries/areas.'
        if len(region_info_count) - 1 > 2:
            region_count_out = 'These documents come from more than ' + str(len(region_info_count) - 1) + ' countries/areas.'
        elif len(region_info_count) - 1 <= 2:
            out = ''
            for elm in region_info_count:
                if not elm[0] == 'Unknown':
                    out = out + elm[0] + ', '
            out = out + 'and other counties/areas.'
            region_count_out = 'These documents come from ' + out
    else:
        if len(region_info_count) > 3:
            region_count_out = 'These documents come from more than ' + str(len(region_info_count)) + ' countries/areas.'
        elif len(region_info_count) <= 3:
            out = ''
            for i, elm in enumerate(region_info_count):
                if not i == len(region_info_count) - 1:
                    out = out + elm[0] + ', '
                else: out = out + 'and ' + elm[0] + '.'
            region_count_out = 'These documents come from ' + out

    print('')
    print(region_count_out)


    # Identify outliers in count for geo figure.
    geo_count_outliers = mad_based_outlier(info_count[:, 0])
    tgt_info_count = info_count[geo_count_outliers, :]
    ind = np.where(tgt_info_count[:, 0] > np.mean(info_count[:, 0]))
    tgt_info_count = tgt_info_count[ind[0], :]

    for i, ab in np.ndenumerate(tgt_info_count[:,2]):
        if region_info_count[int(ab)][0] == 'Unknown':   # ignore the Unknown countries/area.
            tgt_info_count = np.delete(tgt_info_count, (i), axis=0)
            break

    if len(region_info_count) > 2 or (len(region_info_count) == 2 and not unknown):
        if tgt_info_count.shape[0] == 0:
            top_count = 'The documents come evenly from these countries/areas, with mean of ' + '{:.1%}'.format(np.mean(info_count[:, 0])) + ' and SD of ' + '{:.2%}'.format(np.std(info_count[:, 0])) + '.'
            print(propout + top_count)
        elif tgt_info_count.shape[0] == 1:
            if not region_info_count[int(tgt_info_count[0][2])][0] == 'Unknown':
                top_count = region_info_count[int(tgt_info_count[0][2])][0] + ' captures comparably more documents ' + ' (' + '{:.1%}'.format(info_count[int(tgt_info_count[0][2])][0]) + ') ' + ' than the others. '
            #else:
                #top_count = 'The source countries/areas of the documents cannot be identified.'
            print(top_count)
        elif tgt_info_count.shape[0] > 1:
            top_count = 'The following countries capture comparably more documents than the others: '
            print(top_count)
            for i, ab in np.ndenumerate(tgt_info_count):
                propout = region_info_count[int(tgt_info_count[i][2])][0] + ': ' + ' (' + '{:.1%}'.format(info_count[int(tgt_info_count[i][2])][0]) + ').'
                print(propout)

    print('')

    # Identify outliers in sentiment for geo figure.
    geo_sent_outliers = mad_based_outlier(info_sent[:, 1])
    tgt_info_sent = info_sent[geo_sent_outliers, :]

    for i, ab in np.ndenumerate(tgt_info_sent[:, 2]):
        if region_info_sent[int(ab)][0] == 'Unknown':
            tgt_info_sent = np.delete(tgt_info_sent, (i), axis=0)
            break

    ind = np.where(tgt_info_sent[:, 1] > np.mean(info_sent[:, 1]))
    tgt_info_pos_sent = tgt_info_sent[ind[0], :]
    ind = np.where(tgt_info_sent[:, 1] < np.mean(info_sent[:, 1]))
    tgt_info_neg_sent = tgt_info_sent[ind[0], :]



    if len(region_info_count) > 2 or (len(region_info_count) == 2 and not unknown):
        if tgt_info_pos_sent.shape[0] == 0 and tgt_info_neg_sent.shape[0] == 0:
            top_count = 'The sentiment scores of these documents are similar across different countries/areas, with mean of ' + '{:.2f}'.format(np.mean(info_sent[:, 1])) + ' and SD of ' + '{:.2f}'.format(np.std(info_sent[:, 1]))+ '.'
            print(propout + top_count)
        elif tgt_info_pos_sent.shape[0] == 1:
            if not region_info_count[int(tgt_info_pos_sent[0][2])][0] == 'Unknown':
                top_count = region_info_sent[int(tgt_info_pos_sent[0][2])][
                                0] + ' captures comparably more positive documents than the others:' + ' On average ' + '{:.2f}'.format(
                    info_sent[int(tgt_info_pos_sent[0][2])][1]) + ' out of a (-5, +5) scale over ' + str(int(tgt_info_pos_sent[0][0])) + ' Documents ('+\
                            '{:.2%}'.format(int(tgt_info_pos_sent[0][0]) / total_count) + ' in total.)'
            print(top_count)
        elif tgt_info_pos_sent.shape[0] > 1:
            top_count = 'The following countries capture comparably more positive documents than the others: '
            print(top_count)
            for i, ab in np.ndenumerate(tgt_info_pos_sent[:,0]):
                propout = region_info_sent[int(tgt_info_pos_sent[i][2])][0] + ': ' + ' On average ' + '{:.2f}'.format(
                    info_sent[int(tgt_info_pos_sent[i][2])][1]) + ' out of a (-5, +5) scale over ' + str(int(tgt_info_pos_sent[i][0])) + ' Documents ('+\
                            '{:.2%}'.format(int(tgt_info_pos_sent[i][0]) / total_count) + ' in total.)'
                print(propout)

    if len(region_info_count) > 2 or (len(region_info_count) == 2 and not unknown):
        if tgt_info_neg_sent.shape[0] == 1:
            if not region_info_count[int(tgt_info_neg_sent[0][2])][0] == 'Unknown':
                top_count = region_info_sent[int(tgt_info_neg_sent[0][2])][
                                0] + ' captures comparably more negative documents than others:' + ' On average ' + '{:.2f}'.format(
                    info_sent[int(tgt_info_neg_sent[0][2])][1]) + ' out of a (-5, +5) scale over ' + str(int(tgt_info_neg_sent[0][0])) + ' Documents ('+\
                            '{:.2%}'.format(int(tgt_info_neg_sent[0][0]) / total_count) + ' in total.)'

            print(top_count)
        elif tgt_info_neg_sent.shape[0] > 1:
            top_count = 'The following countries capture comparably more negative documents than the others: '
            print(top_count)
            for i, ab in np.ndenumerate(tgt_info_neg_sent[:, 0]):
                propout = region_info_sent[int(tgt_info_neg_sent[i][2])][0] + ': On average ' + '{:.2f}'.format(
                    info_sent[int(tgt_info_neg_sent[i][2])][1]) + ' out of a (-5, +5) scale over ' + str(int(tgt_info_neg_sent[i][0])) + ' Documents ('+\
                            '{:.2%}'.format(int(tgt_info_neg_sent[i][0]) / total_count) + ' in total.)'
                print(propout)

    print('')

    """
    Time trend analysis
    """


    temp_trend(tpcnum, slt, start, end, interval, dict_tpc)
