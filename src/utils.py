# Python modules.
import os
import json
import random
import re
import zipfile


# Other modules.
from kaggle.rest import ApiException
import numpy as np
import pandas as pd


# Functions.
def unzip_file(file_path="data/playground-series-s5e2.zip", output_folder="data/", remove_archive=True) -> None:
    """Unzip file with '.zip' extension.

    :param file_path:
    :param output_folder:
    """
    if not file_path[-3:] == "zip":
        raise ValueError("")
    # Unzip the file
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(output_folder)
    # Cleaning
    if remove_archive:
        os.remove(file_path)


def download_data_from_kaggle_competition(competition_name="playground-series-s5e2", output_folder="data/") -> None:
    """Download files from kaggle coimpetition.

    :param competition_name: Kaggle competition name
    :param output_folder:
    """
    from kaggle.api.kaggle_api_extended import KaggleApi
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Download the specific file
    kaggle_api.competition_download_files(competition=competition_name, path=output_folder)


def submit_file(message: str, submission_file="data/my_submission.csv", competition_name="playground-series-s5e2") -> None:
    """Submit file to kaggle competition.

    :param submission_file:
    :param competition_name: Kaggle competition name
    :note: 
    - get reason pleases
    """
    print(f"Submitting {submission_file} to competition:{competition_name}")
    from kaggle.api.kaggle_api_extended import KaggleApi
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()
    try:
        kaggle_api.competition_submit(file_name=submission_file, message=message, competition=competition_name)
    except ApiException as exc:
        match = re.search(r'HTTP response body: (\{.*\})', str(exc))
        if match:
            json_body = match.group(1)
            try:
                data = json.loads(json_body)
                message = data.get("message", "")
                
                # Extract "Submission not allowed" message
                if "Submission not allowed" in message:
                    hours_match = re.search(r'(\d+\.\d+) hours', message)
                    hours_left = hours_match.group(1) if hours_match else None
                    print(f"Message: Submission not allowed.")
                    if hours_left:
                        print(f"Time remaining: {hours_left} hours.")
            except json.JSONDecodeError:
                raise exc
        else:
            raise exc


def get_submission_scores(competition_name="playground-series-s5e2") -> None:
    """Retrieve kaggle competition scores.

    :param competition_name: Kaggle competition name
    """
    print(f"Retrieving scores from competition:{competition_name}")
    from kaggle.api.kaggle_api_extended import KaggleApi
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()
    submissions = kaggle_api.competition_submissions(competition_name)
    for submission in submissions:
        submission_id = submission.ref
        public_score = submission.publicScore
        private_score = submission.privateScore
        date = submission.date
        print(f"{submission_id} - {public_score} - {private_score} - {date}")


def generate_random_prediction_file(input_path="data/sample_submission.csv", output_filepath="data/my_submission.csv") -> None:
    """Generate a random file of prediction.

    :parm input_path:
    :param output_path:
    """
    df = pd.read_csv(filepath_or_buffer=input_path, sep=",")
    df["Price"] = df["Price"].apply(lambda x: random.uniform(60.0, 100.0))
    print(f"File {output_filepath} successfully generated.\n")
    df.to_csv(output_filepath, sep=",", index=False)


def prepare_submission(
        df_predictions: pd.DataFrame,
        submission_file="data/sample_submission.csv",
        output_path="data/my_submission.csv",
 ) -> pd.DataFrame:
    """

    :param df:
    :param submission_file:
    :param output_path:
    """
    df_sub = pd.read_csv(submission_file, sep=",")
    df_final = pd.merge(
        df_sub,
        df_predictions,
        on="id",
        how="left",
    )[["id", "prediction"]]
    df_final = df_final.rename(columns={"prediction": "Price"})
    df_final["Price"] = df_final.Price.fillna(value=80.0)
    print(f"Generating output: {output_path}.")
    df_final.to_csv(output_path, sep=",", index=False, float_format="%.2f")


def make_a_sound(message='training finished') -> None:
    """Just make a sound in order to know(for example) when training is finished.

    :param message:
    """
    os.system(f"say '{message}.'")
