"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

#from calendar import c
import math
#from tkinter.tix import INTEGER

def viterbi_1(train, test):
        '''
        input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
                test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
        output: list of sentences, each sentence is a list of (word,tag) pairs.
                E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
        '''
        # Variables for smoothing:
        # V: unique words seen in training data for tag T
        tag_unique_words = {} # tag: {word: word_count}
        # n: total number of words in training data for tag T
        tag_total_words = {}
        # V: unique curr tags seen in training data for previous tag T
        tag_unique_tags = {} # prev_tag: {curr_tag: tag_count}
        # n: total number of curr tags in training data for previous tag T
        tag_total_tags = {}
        #alpha: laplace constant
        LAPLACE = 0.00001
        #all tags seen:
        all_tags = set()

        #total sentences:
        n_sen = 0
        #start tag probability
        st_tag_ct = {}
     
        for sentence in train:
                n_sen += 1 # increment number of sentences
                sentence.pop(0) # remove START
                sentence.pop() # remove END
                for i in range(len(sentence)):
                        cpair = sentence[i]
                        # add tag to all tags set:
                        all_tags.add(cpair[1])
                        # for smoothing:
                        #1. word emission smoothing
                        if cpair[1] not in tag_unique_words: # have not seen this tag
                                tag_unique_words[cpair[1]] = {}
                                tag_total_words[cpair[1]] = 0
                        tag_total_words[cpair[1]] += 1 # add one to total words
                        #add word to unique words:
                        if cpair[0] not in tag_unique_words[cpair[1]]:
                                tag_unique_words[cpair[1]][cpair[0]] = 1
                        else:
                                tag_unique_words[cpair[1]][cpair[0]] += 1
                        #2. tag transition smoothing
                        if i > 0:
                                prev_tag = sentence[i-1][1]
                                if prev_tag not in tag_unique_tags:
                                        tag_unique_tags[prev_tag] = {}
                                        tag_total_tags[prev_tag] = 0
                                tag_total_tags[prev_tag] += 1
                                # add curr tag to unique tags for prev tag:
                                if cpair[1] not in tag_unique_tags[prev_tag]:
                                        tag_unique_tags[prev_tag][cpair[1]] = 1
                                else:
                                        tag_unique_tags[prev_tag][cpair[1]] += 1
                                
                        #for training
                        #1. find initial tag probability
                        if i == 0:
                                if cpair[1] not in st_tag_ct:
                                        st_tag_ct[cpair[1]] = 1
                                else:
                                        st_tag_ct[cpair[1]] += 1
        all_tags = list(all_tags)            
        #Decoding test set:
        output = []
        ct = 0
        for sentence in test: 
                
                sentence.pop(0) # remove START
                sentence.pop() # remove END
                #trellis: 2d array: nrows = no. of tags. ncols = no of words in sentence
                trellis = [[-100000000000000000000000 for _ in range(len(sentence))] for _ in range(len(all_tags))]
                #backtracking dictionary: key: (row, col) position of curr node, value is (row, col) position of previous node that gives best probability
                parents_list = {}
                for i in range(len(sentence)):
                        cword = sentence[i]
                        #if this is a starting word
                        if i == 0:
                                #find start tag prob
                                for j in range(len(all_tags)):
                                        #find prob that this tag is starting tag (apply log)
                                        if all_tags[j] not in st_tag_ct:
                                                is_init_prob = LAPLACE / (n_sen + LAPLACE * (len(st_tag_ct) + 1))
                                        else:     
                                                is_init_prob = (st_tag_ct[all_tags[j]] + LAPLACE) / (n_sen + LAPLACE * (len(st_tag_ct) + 1))
                                        is_init_prob_lg = math.log(is_init_prob,2)
                                        #find prob thatcurr word is emitted by this tag
                                        #if this tag has never emitted this word
                                        if cword not in tag_unique_words[all_tags[j]]:
                                                em_prob = LAPLACE / (tag_total_words[all_tags[j]] + LAPLACE * (len(tag_unique_words[all_tags[j]]) + 1))
                                                #print(em_prob)
                                        else:
                                                em_prob = (tag_unique_words[all_tags[j]][cword] + LAPLACE) / (tag_total_words[all_tags[j]] + LAPLACE * (len(tag_unique_words[all_tags[j]]) + 1))
                                        #get log of emission probability
                                        em_prob_lg = math.log(em_prob, 2)
                                        curr_lg_prob = is_init_prob_lg + em_prob_lg
                                        trellis[j][i] = curr_lg_prob
                                        parents_list[(j,i)] = None
                        # for subsequent words (i > 0)
                        else:
                                # for each possible current tag
                                for j in range(len(all_tags)):
                                        #for the current tag, see which of the previous tags will give best probability:
                                        curr_max_lg_prob = -100000000000000000000000
                                        curr_best_parent_tag = None #(i, j)
                                        for pj in range(len(all_tags)):
                                                #find prob that this tag transitioned from previous tag
                                                if all_tags[j] not in tag_unique_tags[all_tags[pj]]: #1. if the previous tag has never transitioned this current tag
                                                        trans_prob = LAPLACE / (tag_total_tags[all_tags[pj]] + LAPLACE * (len(tag_unique_tags[all_tags[pj]]) + 1))
                                                else:
                                                        trans_prob = (tag_unique_tags[all_tags[pj]][all_tags[j]] + LAPLACE) / (tag_total_tags[all_tags[pj]] + LAPLACE * (len(tag_unique_tags[all_tags[pj]]) + 1))
                                                trans_prob_lg = math.log(trans_prob, 2)                                      
                                                
                                                #find prob thatcurr word is emitted by this tag
                                                if cword not in tag_unique_words[all_tags[j]]: #if this tag has never emitted this word
                                                        em_prob = LAPLACE / (tag_total_words[all_tags[j]] + LAPLACE * (len(tag_unique_words[all_tags[j]]) + 1))
                                                else:
                                                        em_prob = (tag_unique_words[all_tags[j]][cword] + LAPLACE) / (tag_total_words[all_tags[j]] + LAPLACE * (len(tag_unique_words[all_tags[j]]) + 1))
                                                #get log of emission probability
                                                em_prob_lg = math.log(em_prob, 2)
                                                #curr path is the best path prob for the current parent + the probability of this current node
                                                curr_lg_prob = trellis[pj][i-1] + trans_prob_lg + em_prob_lg
                                                #find if the current tag is the best tag
                                                #print(curr_lg_prob)
                                                if curr_lg_prob > curr_max_lg_prob:
                                                        curr_best_parent_tag = pj
                                                        curr_max_lg_prob = curr_lg_prob
                                        # print('best prob for this tag: ', curr_max_lg_prob)
                                        trellis[j][i] = curr_max_lg_prob
                                        parents_list[(j,i)] = (curr_best_parent_tag, i-1)
                                        
                ### evaluating the TRELLIS (find best path) and adding into processed sentence
                proc_sentence = []
                #get best position at the last stage:
                max_prob_last_stg = -100000000000000000000000
                best_idx_last_stg = -1
                for idx in range(len(all_tags)):
                        if trellis[idx][-1] > max_prob_last_stg:
                                max_prob_last_stg = trellis[idx][-1]
                                best_idx_last_stg = idx
                # put the last tag of best path into proc_sentence
                proc_sentence.append((sentence[-1], all_tags[best_idx_last_stg]))
                # put all subsequent tags in best path into proc_sentence by backtracking
                next_best = parents_list[(best_idx_last_stg, len(sentence)-1)]
                while next_best: # next best is a pair (tag_idx, word_idx)
                        #print(next_best, all_tags[next_best[0]])
                        curr_tagging = (sentence[next_best[1]], all_tags[next_best[0]])
                        proc_sentence.append(curr_tagging)
                        
                        next_best = parents_list[(next_best[0], next_best[1])]
                
                proc_sentence.append(('START', 'START'))
                proc_sentence.insert(0,('END','END'))
                output.append(proc_sentence[::-1])
                # if ct < 3:
                #         print(sentence)
                #         print(proc_sentence[::-1])
                # ct += 1
        return output
                                

        