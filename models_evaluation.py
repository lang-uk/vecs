import argparse
import logging
import pathlib
import gensim
import itertools
import csv
import glob
import traceback
from os import path
from collections import defaultdict

logger = logging.getLogger("fasttext")


def questions_reading(analogies):
    """
    Reads all type of questions to dict.
    :param analogies: string
    :return: dict
    """
    analogy_dict = defaultdict(list)
    with open(analogies, "r") as analogy_f:
        key = None
        for line in map(str.strip, analogy_f):
            if not line:
                continue
            if line.startswith(":"):
                # new section starts
                key = line
            else:
                analogy_dict[key].append(line.split("\t"))
    return analogy_dict


def analogy_evaluation(models_path, analogies, first_n, file_out):
    """
    Compute performance of the model on an analogy test set and writes results to a file.
    :param models_path: string
    :param analogies: dict
    :param first_n: int
    :param file_out: string
    :return: None
    """
    if path.exists(file_out):
        csv_file = open(file_out, 'a')
        writer = csv.writer(csv_file)
    else:
        csv_file = open(file_out, 'w')
        writer = csv.writer(csv_file)
        header = ["Model", "Type of questions", "Accuracy1 (for present questions)", "Accuracy2 (for all questions)",
                  "Number of questions", "Number of present questions", "No words in vocabulary"]
        writer.writerow(header)

    for file in glob.glob(models_path):
        if not path.isfile(file):
            continue
        name = path.basename(file).replace(".", "_")
        logger.info(f"Model {name} loading...")

        # TODO: Ideally here we should check the file extension and load binary fb vectors or textual w2v
        model = gensim.models.fasttext.load_facebook_vectors(file)

        logger.info("-------------------------------------------------------")
        logger.info("Starting evaluation")

        total_analogies = len(list((itertools.chain(*analogies.values()))))

        total_correct = 0
        present_analogies = 0

        for key, questions in analogies.items():
            section_correct = 0
            total_section = len(questions)
            present_analogies += total_section
            for i in range(len(questions)):
                try:
                    sim = list(zip(*(
                        model.most_similar(positive=[questions[i][0], questions[i][3]],
                                           negative=[questions[i][2]],
                                           topn=first_n))))[0]
                    if questions[i][1].lower() in [s.lower() for s in sim]:
                        section_correct += 1
                        total_correct += 1
                    else:
                        sim = list(zip(*(
                            model.most_similar(positive=[questions[i][0].lower(), questions[i][3].lower()],
                                               negative=[questions[i][2].lower()],
                                               topn=first_n))))[0]

                        if questions[i][1].lower() in sim or questions[i][1].lower() in [s.lower() for s in sim]:
                            section_correct += 1
                            total_correct += 1

                except Exception:
                    logging.error(traceback.format_exc())
                    total_section -= 1

            # calculate and write to file accuracy for each section
            if total_section != 0:
                acc, acc_general = log_evaluate_model(section_correct, total_section, len(questions), key)
                writer.writerow([name, key.split(" ")[1], acc, acc_general, len(questions), total_section, ""])
                csv_file.flush()
            else:
                logger.info(f"There are no words in vocabulary for type of questions = {key}")
                writer.writerow([name, key.split(" ")[1], "", "", len(questions), total_section, "+"])
                csv_file.flush()

        # calculate and write to file total accuracy
        if present_analogies != 0:
            acc, acc_general = log_evaluate_model(total_correct, present_analogies, total_analogies, "total")
            writer.writerow([name, "total", acc, acc_general, total_analogies, present_analogies, ""])
            csv_file.flush()
        else:
            logger.info("There are no words in vocabulary for all questions.")

    csv_file.close()


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
                        default=pathlib.Path("test/test_vocabulary.txt"))
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

    # model testing and results saving
    analogy_evaluation(args.models_path, category_questions, int(args.first_n), args.results)
