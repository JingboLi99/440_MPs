"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
        '''
        input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
                test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
        output: list of sentences, each sentence is a list of (word,tag) pairs.
                E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
        '''
        #wordtags tracks the tags for a given word: key-> word | value-> dictionary key: tag, value: count
        wordtags = {}
        total_tag = {}
        for sentence in train:
                for pair in sentence:
                        if pair[0] not in wordtags:
                                wordtags[pair[0]] = {}
                        if pair[1] not in total_tag:
                                total_tag[pair[1]] = 1
                        else:
                                total_tag[pair[1]] += 1
                                
                        if pair[1] not in wordtags[pair[0]]:
                                wordtags[pair[0]][pair[1]] = 1
                        else:
                                wordtags[pair[0]][pair[1]] += 1
        #evaluate most likely tag for each word
        proc_wordtags = {}
        for word, tags in wordtags.items():
                best_tag = None
                tag_ct = 0
                for t, ct in tags.items():
                        if ct > tag_ct:
                                tag_ct = ct
                                best_tag = t
                proc_wordtags[word] = best_tag
        #evalualte most likely tag in total
        best_tag = None
        best_tag_ct = 0
        for tag, ct in total_tag.items():
                if ct > best_tag_ct:
                        best_tag_ct = ct
                        best_tag = tag
        
                                
        output = []
        for sentence in test:
                tagged_sent = []
                for word in sentence:
                        if word in proc_wordtags:
                                tagged_sent.append((word, proc_wordtags[word]))
                        else:
                                tagged_sent.append((word, best_tag))
                output.append(tagged_sent)
        return output