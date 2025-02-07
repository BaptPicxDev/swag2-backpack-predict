# Pyhton modules

# Other modules

# Librairies
from src.utils import (
    download_data_from_kaggle_competition,
    unzip_file,
    generate_random_prediction_file,
    submit_file,
    get_submission_scores,
)

# Main thread
if __name__ == "__main__":
    download_data_from_kaggle_competition()
    unzip_file(),
    generate_random_prediction_file()
    submit_file(message="random test")
    get_submission_scores()
