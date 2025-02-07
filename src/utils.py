# Python modules.
import os
import zipfile


# Functions
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
    """
    print(f"Submitting {submission_file} to competition:{competition_name}")
    from kaggle.api.kaggle_api_extended import KaggleApi
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()
    kaggle_api.competition_submit(file_name=submission_file, message=message, competition=competition_name)


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
