# -*- coding: utf-8 -*-
import os

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_file_path, output_file_path):
    """ Runs data processing scripts to turn raw datas from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    logger.info('making final data set from raw data ' + input_file_path + " to processed data " + output_file_path)


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv(project_dir))
    input_file_path = os.getenv("Output_filepath")
    output_file_path = os.path.join(project_dir, str(os.getenv("Output_filepath")))
    main(input_file_path, output_file_path)
