from os import listdir
from os.path import isfile, join
import sys
import argparse
import logging
import pathlib

import gensim
import pandas as pd
import itertools

logger = logging.getLogger("fasttext")


def questions_reading(analogies):
    """
    Reads all type of questions to dict.
    :param analogies: string
    :return: dict
    """
    analogy_dict = dict()
    with open(analogies, "r") as analogy_f:
        key = None
        for line in analogy_f:
            if line in ['\n', '\r\n']:
                continue
            if line.startswith(":"):
                # new section starts
                key = line.rstrip()
                analogy_dict[key] = list()
            else:
                analogy_dict[key].append(line[:-1].split("\t"))
    return analogy_dict


def analogy_evaluation(models_path, analogies, first_n):
    """
    Compute performance of the model on an analogy test set.
    :param models_path: string
    :param analogies: dict
    :param first_n: int
    :return: DataFrame
    """
    result = pd.DataFrame(
        columns=["Model", "Type of questions", "Accuracy1 (for present questions)", "Accuracy2 (for all questions)",
                 "Number of questions", "Number of present questions", "No words in vocabulary"])

    for file in listdir(models_path):
        if not isfile(join(models_path, file)):
            continue
        name = file.replace(".", "_")
        logger.info(f"Model {name} loading...")

        # TODO: Ideally here we should check the file extension and load binary fb vectors or textual w2v
        model = gensim.models.fasttext.load_facebook_vectors(models_path + file)

        logger.info("-------------------------------------------------------")
        logger.info("Starting evaluation")

        total_analogies = len(list((itertools.chain(*analogies.values()))))

        total_correct = 0
        present_analogies = 0

        for key, questions in analogies.items():
            section_correct = 0
            total_section = len(questions)
            present_analogies += total_section
            # TODO: validate how many tasks for each category do we have?
            for i in range(len(questions)):
                try:
                    sim = list(list(zip(*(
                        model.most_similar(positive=[questions[i][0], questions[i][3]], negative=[questions[i][2]],
                                           topn=first_n))))[0])
                    if (questions[i][1] in sim
                            or questions[i][1].lower() in sim
                            or questions[i][1] in [s.lower() for s in sim]
                            or questions[i][1].lower() in [s.lower() for s in sim]):
                        section_correct += 1
                        total_correct += 1
                    else:
                        try:
                            # TODO: double check if that contributes any boost to the score on our vectors
                            #       If not: make it optional and disabled by default
                            sim = list(list(zip(*(
                                model.most_similar(positive=[questions[i][0].lower(), questions[i][3].lower()],
                                                   negative=[questions[i][2].lower()],
                                                   topn=first_n))))[0])

                            if questions[i][1].lower() in sim or questions[i][1].lower() in [s.lower() for s in sim]:
                                section_correct += 1
                                total_correct += 1

                            else:
                                pass

                        except:
                            total_section = total_section - 1

                except:
                    try:
                        # TODO: verify when and how often does it happen
                        sim = list(list(zip(*(
                            model.most_similar(positive=[questions[i][0].lower(), questions[i][3].lower()],
                                               negative=[questions[i][2].lower()],
                                               topn=first_n))))[0])
                        if questions[i][1].lower() in sim or questions[i][1].lower() in [s.lower() for s in sim]:
                            section_correct += 1
                            total_correct += 1
                        else:
                            pass

                    except:
                        total_section = total_section - 1
            # calculate and write to DataFrame accuracy for each section
            if total_section != 0:
                acc, acc_general = log_evaluate_model(section_correct, total_section, len(questions), key)
                result.loc[len(result)] = [name, key.split(" ")[1], acc, acc_general, len(questions), total_section, ""]
            else:
                logger.info(f"There are no words in vocabulary for type of questions = {key}")
                result.loc[len(result)] = [name, key.split(" ")[1], "", "", len(questions), total_section, "+"]

        # calculate and write to DataFrame total accuracy
        if present_analogies != 0:
            acc, acc_general = log_evaluate_model(total_correct, present_analogies, total_analogies, "total")
            result.loc[len(result)] = [name, "total", acc, acc_general, total_analogies, present_analogies, ""]
        else:
            logger.info("There are no words in vocabulary for all questions.")
    return result


def log_evaluate_model(correct, total_present, total_general, section_name):
    """
    Computes model accuracy (number of correct predictions divided by number of all questions) and logs result.
    :param correct: int
    :param total_present: int
    :param total_general: int
    :param section_name: string
    :return: float, float
    """
    acc = (correct / total_present) * 100.0
    # TODO: does it really check whether questions are in vocabulary ?
    acc_general = (correct / total_general) * 100.0
    logger.info(
        f"Type of question = {section_name},   Accuracy1  = {acc}    (if question's words are in vocabulary)")
    logger.info(f"Type of question = {section_name},   Accuracy2  = {acc_general}    (for all questions)")
    return acc, acc_general


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform intrinsic evaluation of word vectors using gensim. You can evaluate more than one model"
    )
    parser.add_argument("--questions", type=pathlib.Path, help="Path to the file with test questions",
                        default=pathlib.Path("test/test_vocabulary.py"))
    parser.add_argument("--first_n", type=int, help="Number of variants to look into", default=4)
    parser.add_argument("models_path", type=str, help="Path to models for validation")
    parser.add_argument("results", type=pathlib.Path, help="File to store results too (csv)")
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        choices=(0, 1, 2, 3),
        help="Level of verbosity",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # questions reading and parsing
    category_questions = questions_reading(args.questions)

    # model testing
    df_output = analogy_evaluation(args.models_path, category_questions, int(args.first_n))

    # results saving
    df_output.to_csv(args.results, index=False)
