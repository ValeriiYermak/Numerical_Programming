1.
# def  bayes_theorem( p_a, p_b_given_a, p_b_given_not_a ):
#     """
#     Обчисліть апостеріорну ймовірність за допомогою теореми Баєса.
#
#     :param p_a: Попередня ймовірність A
#     :param p_b_given_a: Імовірність B, якщо A
#     :param p_b_given_not_a: Імовірність B, якщо не A
#     : return: обчислена ймовірність
#     """
#     p_not_a = 1 - p_a
#     p_b = (p_b_given_a * p_a) + (p_b_given_not_a * p_not_a)
#     return (p_b_given_a * p_a) / p_b
#
# # Присвоєння значень на основі нашої проблеми
# p_disease = 0.0001
# p_positive_given_disease = 0.99
# p_positive_given_no_disease = 0.01
#
# # Розрахунок імовірності захворювання при позитивному результаті тесту
# p_disease_given_positive = bayes_theorem(p_disease, p_positive_given_disease, p_positive_given_no_disease)
# print(p_disease_given_positive)

2.

# train_spam = ['send us your password', 'review our website', 'send your password', 'send us your account']
# train_ham = ['Your activity report','benefits physical activity', 'the importance vows']
# test_emails = {'spam':['renew your password', 'renew your vows'], 'ham':['benefits of our account', 'the importance of physical activity']}
#
# # make a vocabulary of unique words that occur in known spam emails
#
# vocab_words_spam = []
#
# for sentence in train_spam:
#     sentence_as_list = sentence.split()
#     for word in sentence_as_list:
#         vocab_words_spam.append(word)
#
# print(vocab_words_spam)
#
# vocab_unique_words_spam = list(set(vocab_words_spam))
# print(vocab_unique_words_spam)
#
# dict_spamicity = {}
# for w in vocab_unique_words_spam:
#     emails_with_w = 0     # counter
#     for sentence in train_spam:
#         if w in sentence:
#             emails_with_w+=1
#
#     print(f"Number of spam emails with the word '{w}': {emails_with_w}")
#     total_spam = len(train_spam)
#     spamicity = (emails_with_w+1)/(total_spam+2)
#     print(f"Spamicity of the word '{w}': {spamicity} \n")
#     dict_spamicity[w.lower()] = spamicity
#
# # make a vocabulary of unique words that occur in known ham emails
# vocab_words_ham = []
# for sentence in train_ham:
#     sentence_as_list = sentence.split()
#     for word in sentence_as_list:
#         vocab_words_ham.append(word)
#
# vocab_unique_words_ham = list(set(vocab_words_ham))
# print(vocab_unique_words_ham)
#
# dict_hamicity = {}
# for w in vocab_unique_words_ham:
#     emails_with_w = 0     # counter
#     for sentence in train_ham:
#         if w in sentence:
#             print(w+":", sentence)
#             emails_with_w+=1
#
#     print(f"Number of ham emails with the word '{w}': {emails_with_w}")
#     total_ham = len(train_ham)
#     Hamicity = (emails_with_w+1)/(total_ham+2)       # Smoothing applied
#     print()
#     print(f"Hamicity of the word '{w}': {Hamicity} ")
#     dict_hamicity[w.lower()] = Hamicity
#                                          # Use built-in lower() to keep all words lower case - useful later when
#                                          # comparing spamicity vs hamicity of a single word - e.g. 'Your' and
#                                          # 'your' will be treated as 2 different words if not normalized to lower                                          # case.
#
# print(dict_hamicity)
#
# prob_spam = len(train_spam) / (len(train_spam)+(len(train_ham)))
# print(prob_spam)
#
# prob_ham = len(train_ham) / (len(train_spam)+(len(train_ham)))
# print(prob_ham)
#
# tests = []
# for i in test_emails['spam']:
#     tests.append(i)
#
# for i in test_emails['ham']:
#     tests.append(i)
#
# print(tests)
#
# # split emails into distinct words
#
# distinct_words_as_sentences_test = []
#
# for sentence in tests:
#     sentence_as_list = sentence.split()
#     senten = []
#     for word in sentence_as_list:
#         senten.append(word)
#     distinct_words_as_sentences_test.append(senten)
#
# print(distinct_words_as_sentences_test)
#
# test_spam_tokenized = [distinct_words_as_sentences_test[0], distinct_words_as_sentences_test[1]]
# test_ham_tokenized = [distinct_words_as_sentences_test[2], distinct_words_as_sentences_test[3]]
# print(test_spam_tokenized)
#
# reduced_sentences_spam_test = []
# for sentence in test_spam_tokenized:
#     words_ = []
#     for word in sentence:
#         if word in vocab_unique_words_spam:
#             print(f"'{word}', ok")
#             words_.append(word)
#         elif word in vocab_unique_words_ham:
#             print(f"'{word}', ok")
#             words_.append(word)
#         else:
#             print(f"'{word}', word not present in labelled spam training data")
#     reduced_sentences_spam_test.append(words_)
# print(reduced_sentences_spam_test)
# reduced_sentences_ham_test = []                   # repeat for ham words
# for sentence in test_ham_tokenized:
#     words_ = []
#     for word in sentence:
#         if word in vocab_unique_words_ham:
#             print(f"'{word}', ok")
#             words_.append(word)
#         elif word in vocab_unique_words_spam:
#             print(f"'{word}', ok")
#             words_.append(word)
#         else:
#             print()
#             print(f"'{word}', word not present in labelled ham training data")
#     reduced_sentences_ham_test.append(words_)
# print()
# print(reduced_sentences_ham_test)
#
#
# test_spam_stemmed = []
# non_key = ['us',  'the', 'of','your']       # non-key words, gathered from spam,ham and test sentences
# for email in reduced_sentences_spam_test:
#     email_stemmed=[]
#     for word in email:
#         if word in non_key:
#             print('remove')
#         else:
#             email_stemmed.append(word)
#     test_spam_stemmed.append(email_stemmed)
#
# print(test_spam_stemmed)
#
# test_ham_stemmed = []
# non_key = ['us',  'the', 'of', 'your']
# for email in reduced_sentences_ham_test:
#     email_stemmed=[]
#     for word in email:
#         if word in non_key:
#             print('remove')
#         else:
#             email_stemmed.append(word)
#     test_ham_stemmed.append(email_stemmed)
#
# print(test_ham_stemmed)
# print(len(reduced_sentences_ham_test))


3.

def mult(list_) :        # function to multiply all word probs together
    total_prob = 1
    for i in list_:
         total_prob = total_prob * i
    return total_prob

def Bayes(email):
    probs_s = []
    probs_h = []
    for word in email:
        Pr_S = prob_spam
        # print('prob of spam in general ',Pr_S)
        try:
            pr_WS = dict_spamicity[word]
            # print(f'prob "{word}"  is a spam word : {pr_WS}')
        except KeyError:
            pr_WS = 1/(total_spam+2)  # Apply smoothing for word not seen in spam training data, but seen in ham training
            # print(f"prob '{word}' is a spam word: {pr_WS}")

        Pr_H = prob_ham
        # print('prob of ham in general ', Pr_H)
        try:
            pr_WH = dict_hamicity[word]
            # print(f'prob "{word}" is a ham word: ',pr_WH)
        except KeyError:
            pr_WH = (1/(total_ham+2))  # Apply smoothing for word not seen in ham training data, but seen in spam training
            # print(f"WH for {word} is {pr_WH}")
            # print(f"prob '{word}' is a ham word: {pr_WH}")

        prob_word_is_spam_BAYES = pr_WS
        prob_word_is_ham_BAYES = pr_WH


        # print('')
        # print(f"Using Bayes, prob the the word '{word}' is spam: {prob_word_is_spam_BAYES}")
        # print('###########################')
        probs_s.append(prob_word_is_spam_BAYES)
        probs_h.append(prob_word_is_ham_BAYES)
    # print(f"All word probabilities for this sentence: {probs}")
    final_classification = Pr_S*mult(probs_s) /((Pr_S*mult(probs_s))+(Pr_H*mult(probs_h)))
    print('###########################')
    if final_classification >= 0.5:
        print(f'email is SPAM: with spammy confidence of {final_classification*100}%')
    else:
        print(f'email is HAM: with spammy confidence of {final_classification*100}%')
    return final_classification

for email in test_spam_stemmed:
    print('')
    print(f"           Testing stemmed SPAM email {email} :")
    # print('                 Test word by word: ')
    all_word_probs = Bayes(email[0:1])
    print(all_word_probs)


for email in test_ham_stemmed:
    print('')
    print(f"           Testing stemmed HAM email {email} :")
    print('                 Test word by word: ')
    all_word_probs = Bayes(email)
    print(all_word_probs)


