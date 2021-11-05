from os import listdir
from os.path import isfile, join
import sys
import argparse
import logging
import pathlib

import gensim
import pandas as pd


logger = logging.getLogger("fasttext")

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

    # TODO: this is weird and obscure, better code would be hood
    # read all questions
    with open(questions_file, "r") as analogy_f:
        for line in analogy_f:
            questions.append(line)

    for i in range(len(questions)):
        if questions[i].startswith(":"):
            themes.append([questions[i].split("\t")[0], i])

    # split by themes
    category_questions = dict()
    for t in range(len(themes)):
        try:
            category_questions[themes[t][0]] = questions[(themes[t][1] + 1) : themes[t + 1][1]]
        except Exception as e:
            print(e)
            category_questions[themes[t][0]] = questions[(themes[t][1] + 1) :]

    # transform
    all_questions = []
    for k, v in category_questions.items():
        category_questions[k] = [words.split("\n")[0].split("\t") for words in v]
        all_questions = all_questions + [words.split("\n")[0].split("\t") for words in v]

    category_questions[": all questions"] = all_questions
    return category_questions


def model_testing(models_path, category_questions, first_n):
    # define output data frame
    result = pd.DataFrame(
        columns=[
            "Model",
            "Type of questions",
            "Accuracy1 (for present questions)",
            "Accuracy2 (for all questions)",
            "No words in vocabulary",
            "Number of questions",
            "Number of present questions",
        ]
    )

    # read folder with models
    onlyfiles = [f for f in listdir(models_path) if isfile(join(models_path, f))]
    print("Models loading...")
    models = dict()
    for i in onlyfiles:
        # TODO: Ideally here we should check the file extension and load binary fb vectors or textual w2v
        models[i.replace(".", "_")] = gensim.models.fasttext.load_facebook_vectors(models_path + i)

    # TODO: urgent, do not load all at once move that loading to the loop below    
    print("Models evaluation...")

    # for each model
    for name, model in models.items():
        print("-------------------------------------------------------")
        print("-------------------------------------------------------")
        print("-------------------------------------------------------")
        print("Model = ", name)

        # for each question
        for key, questions in category_questions.items():
            n = 0
            m = len(questions)
            # TODO: validate how many tasks for each category do we have?
            for i in range(len(questions)):
                try:
                    # TODO: Do we need a dataframe here?
                    # Things to look at: https://radimrehurek.com/gensim/models/keyedvectors.html#why-use-keyedvectors-instead-of-a-full-model
                    # https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.most_similar
                    # it seems that by default it returns first 10 similar words.
                    # This means, that for the first_n < 10 it's inefficient, for thet first_n > 10 it simply doesn't work correctly (we don't care much but still)
                    # Also most_similar has more params to look into and investigate
                    sim = pd.DataFrame(
                        model.most_similar(
                            positive=[questions[i][0], questions[i][3]],
                            negative=[questions[i][2]],
                        )
                    )[:first_n][0].values

                    if (
                        questions[i][1] in sim
                        or questions[i][1].lower() in sim
                        or questions[i][1] in [s.lower() for s in sim]
                        or questions[i][1].lower() in [s.lower() for s in sim]
                    ):
                        n = n + 1
                    else:
                        try:
                            # TODO: doublecheck if that contributes any boost to the score on our vectors
                            # If not: make it optional and disabled by default
                            sim = pd.DataFrame(
                                model.most_similar(
                                    positive=[
                                        questions[i][0].lower(),
                                        questions[i][3].lower(),
                                    ],
                                    negative=[questions[i][2].lower()],
                                )
                            # TODO: :4 is cleary a bug and should be :first_n
                            )[:4][0].values
                            if questions[i][1].lower() in sim or questions[i][1].lower() in [s.lower() for s in sim]:
                                n = n + 1
                            else:
                                pass

                        except:
                            m = m - 1

                except Exception as e:
                    print(e)
                    try:
                        # TODO: verify when and how often does it happen
                        sim = pd.DataFrame(
                            model.most_similar(
                                positive=[
                                    questions[i][0].lower(),
                                    questions[i][3].lower(),
                                ],
                                negative=[questions[i][2].lower()],
                            )
                        # TODO: same problem as above
                        )[:4][0].values
                        if questions[i][1].lower() in sim or questions[i][1].lower() in [s.lower() for s in sim]:
                            n = n + 1
                        else:
                            pass

                    except:
                        m = m - 1

            if m != 0:
                # calculate accuracy: number of correct answers divided by number of all questions
                acc = (float(n) / float(m)) * 100.0
                acc_general = (float(n) / float(len(questions))) * 100.0
                print(
                    "Type of question = {},   Accuracy1  = {}    (if question's words are in vocabulary)".format(
                        key, acc
                    )
                )
                print("Type of question = {},   Accuracy2  = {}    (for all questions)".format(key, acc_general))
                result.loc[len(result)] = [name, key.split(" ")[1], acc, acc_general, "", len(questions), m]
            else:
                print("There are no words in vocabulary for type of questions = ", key)
                result.loc[len(result)] = [name, key.split(" ")[1], "", "", "+", len(questions), m]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform intrinsic evaluation of word vectors using gensim. You can evaluate more than one model"
    )
    parser.add_argument("--questions", type=pathlib.Path, help="Path to the file with test questions", default=pathlib.Path("test/test_vocabulary.txt"))
    parser.add_argument("--first_n", type=int, help="Number of variants to look into", default=4)
    parser.add_argument("models_path", type=pathlib.Path, help="Path to models for validation")
    parser.add_argument("results", type=pathlib.Path, help="File to store results too (csv)")
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        choices=(0, 1, 2, 3),
        help="Level of verbosity",
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(name)s:%(levelname)s %(asctime)s %(message)s")

    # questions reading and parsing
    category_questions = questions_reading(args.questions)
    # print(category_questions)

#     # model testing
#     df_output = model_testing(models_path, category_questions, int(first_n))

#     # results saving
#     df_output.to_csv(file_output, index=False)
