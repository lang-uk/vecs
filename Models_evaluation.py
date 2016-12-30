
# coding: utf-8

import gensim
import pandas as pd

from os import listdir
from os.path import isfile, join
import sys


# sample how to run
# python Models_evaluation.py ./models_fiction/ test_vocabulary.txt 4 result_fiction.csv

# input parameters
# models_path - path to the folder with models for evaluating
# questions_file - the name of the text file with questions
# first_n - the number of first words in answer, which can be detected as correct answer
# file_output - the csv file for results saving



def questions_reading(questions_file):
    questions = []
    themes = []
    
    # read all questions
    with open(questions_file, "rb") as analogy_f:
            for line in analogy_f:
                questions.append(line)

    for i in xrange(len(questions)):
        if questions[i].startswith(b":"):
            themes.append([questions[i].split("\t")[0], i])

    # split by themes
    category_questions = dict()
    for t in xrange(len(themes)):
        try:
            category_questions[themes[t][0]] = questions[(themes[t][1]+1):themes[t+1][1]]
        except:
            category_questions[themes[t][0]] = questions[(themes[t][1]+1):]

    # transform
    all_questions = []
    for k, v in category_questions.items():
        category_questions[k] = [words.split("\n")[0].split("\t") for words in v]
        all_questions = all_questions + [words.split("\n")[0].split("\t") for words in v]

    category_questions[": all questions"] = all_questions    
    return category_questions
    


def model_testing(models_path, category_questions, first_n):    
    # define output data frame
    result = pd.DataFrame(columns = ["Model", "Type of questions",                                              "Accuracy1 (for present questions)",                                              "Accuracy2 (for all questions)",                                              "No words in vocabulary",                                              "Number of questions",                                             "Number of present questions"])
    
    # read folder with models
    onlyfiles = [f for f in listdir(models_path) if isfile(join(models_path, f))]
    print "Models loading..."
    models = dict()
    for i in onlyfiles:
        models[i.replace(".", "_")] = gensim.models.Word2Vec.load_word2vec_format(models_path+i)   
    models
    print "Models evaluation..."


    # for each model 
    for name, model in models.items():
        print "-------------------------------------------------------"
        print "-------------------------------------------------------"
        print "-------------------------------------------------------"
        print "Model = ", name
        
        # for each question
        for key, questions in category_questions.items():        
            n = 0
            m = len(questions)
            for i in range(len(questions)):
                try:
                    sim = pd.DataFrame(model.most_similar(positive=[unicode(questions[i][0], "utf-8"),                                                                    unicode(questions[i][3], "utf-8")],                                                           negative=[unicode(questions[i][2], "utf-8")]))[:first_n][0].values

                    if unicode(questions[i][1], "utf-8") in sim or                     unicode(questions[i][1].lower(), "utf-8") in sim or                     unicode(questions[i][1], "utf-8") in [s.lower() for s in sim] or                     unicode(questions[i][1].lower(), "utf-8") in [s.lower() for s in sim]:
                        n = n + 1
                    else:
                        try:
                            sim = pd.DataFrame(model.most_similar(positive=[unicode(questions[i][0].lower(), "utf-8"),                                                                        unicode(questions[i][3].lower(), "utf-8")],                                                               negative=[unicode(questions[i][2].lower(), "utf-8")]))[:4][0].values
                            if unicode(questions[i][1].lower(), "utf-8") in sim or                             unicode(questions[i][1].lower(), "utf-8") in [s.lower() for s in sim]:                                
                                n = n + 1
                            else:
                                pass

                        except:   
                            m = m-1

                except:
                    try:
                        sim = pd.DataFrame(model.most_similar(positive=[unicode(questions[i][0].lower(), "utf-8"),                                                                    unicode(questions[i][3].lower(), "utf-8")],                                                           negative=[unicode(questions[i][2].lower(), "utf-8")]))[:4][0].values
                        if unicode(questions[i][1].lower(), "utf-8") in sim or                         unicode(questions[i][1].lower(), "utf-8") in [s.lower() for s in sim]:
                            n = n + 1
                        else:
                            pass

                    except:   
                        m = m-1                   
            

            if m != 0:
                # calculate accuracy: number of correct answers divided by number of all questions
                acc = (float(n)/float(m))*100.0        
                acc_general = (float(n)/float(len(questions)))*100.0  
                print "Type of question = {},   Accuracy1  = {}    (if question's words are in vocabulary)".format(key, acc)
                print "Type of question = {},   Accuracy2  = {}    (for all questions)".format(key, acc_general)
                result.loc[len(result)] = [name, key.split(" ")[1], acc, acc_general, '', len(questions), m]
            else:            
                print "There are no words in vocabulary for type of questions = ", key
                result.loc[len(result)] = [name, key.split(" ")[1], '', '', '+', len(questions), m]
    return result


if __name__ == '__main__':
    # arguments reading    
    models_path = sys.argv[1]
    questions_file = sys.argv[2]
    first_n = sys.argv[3]
    file_output = sys.argv[4]
    
    # questions reading and parsing
    category_questions = questions_reading(questions_file)

    # model testing
    df_output = model_testing(models_path, category_questions, first_n)

    # results saving    
    df_output.to_csv(file_output, index = False)

