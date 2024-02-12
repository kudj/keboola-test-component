"""
Template Component main class.

"""
import csv
import logging
from datetime import datetime

import json
import openai
from typing import Iterator, List, Dict
from csv import DictReader, DictWriter

from keboola.component.base import ComponentBase
from keboola.component.exceptions import UserException


# configuration variables
KEY_API_TOKEN = '#api_token'
KEY_BASE_PROMPT = 'print_hello'
KEY_TEXT_COLUMN = 'text_column'

# list of mandatory parameters => if some is missing,
# component will fail with readable message on initialization.
REQUIRED_PARAMETERS = [KEY_API_TOKEN, KEY_BASE_PROMPT, KEY_TEXT_COLUMN]
REQUIRED_IMAGE_PARS = []

MODEL_NAME = "text-davinci-003"
MODEL_BASE_TEMPERATURE = 0.7
MODEL_BASE_MAX_TOKENS = 512
MODEL_BASE_TOP_P = 1
MODEL_BASE_FREQUENCY_PENALTY = 0
MODEL_BASE_PRESENCE_PENALTY = 0


def read_messages_from_file(file_name: str) -> Iterator[Dict]:
    with open(file_name) as in_file:
        yield from DictReader(in_file)


def process_message(openai_key: str, prompt: str) -> str:
    openai.api_key = openai_key
    response = openai.Completion.create(
        model=MODEL_NAME,
        prompt=prompt,
        temperature=MODEL_BASE_TEMPERATURE,
        max_tokens=MODEL_BASE_MAX_TOKENS,
        top_p=MODEL_BASE_TOP_P,
        frequency_penalty=MODEL_BASE_FREQUENCY_PENALTY,
        presence_penalty=MODEL_BASE_PRESENCE_PENALTY
    )
    return response.choices[0].text


def generate_prompt(base_prompt: str, message: str) -> str:
    return f"{base_prompt}\n\"\"\"{message}\"\"\""


def analyze_messages_in_file(in_file_name: str,
                             text_column: str,
                             out_file_name: str,
                             out_file_columns: List[str],
                             base_prompt: str,
                             openai_key: str) -> None:
    with open(out_file_name, 'w') as out_file:
        writer = DictWriter(out_file, out_file_columns)
        for message in read_messages_from_file(in_file_name):
            prompt = generate_prompt(base_prompt, message.get(text_column))
            data = json.loads(process_message(openai_key, prompt))
            writer.writerow({**message, "open_ai_output": data})


class Component(ComponentBase):
    """
        Extends base class for general Python components. Initializes the CommonInterface
        and performs configuration validation.

        For easier debugging the data folder is picked up by default from `../data` path,
        relative to working directory.

        If `debug` parameter is present in the `config.json`, the default logger is set to verbose DEBUG mode.
    """

    def __init__(self):
        super().__init__()

    def run(self):
        """
        Main execution code
        """

        logging.info("state file")

        now = datetime.now()
        self.write_state_file({"some_state_parameter": now.strftime('%H:%M:%S')})

        params = self.configuration.parameters

        text_column = params.get(KEY_TEXT_COLUMN)
        base_prompt = params.get(KEY_BASE_PROMPT)
        api_token = params.get(KEY_API_TOKEN)

        if base_prompt == "raise_exception":
            raise UserException("This is a user exception")

        input_table = self.get_input_tables_definitions()[0]

        final_columns = input_table.columns + ["open_ai_output"]

        out_table = self.create_out_table_definition("analyzed_output", columns=final_columns)

        analyze_messages_in_file(in_file_name=input_table.full_path,
                                 text_column=text_column,
                                 out_file_name=out_table.full_path,
                                 openai_key=api_token,
                                 out_file_columns=final_columns,
                                 base_prompt=base_prompt)

        self.write_manifest(out_table)


"""
        Main entrypoint
"""
if __name__ == "__main__":
    try:
        comp = Component()
        # this triggers the run method by default and is controlled by the configuration.action parameter
        comp.execute_action()
    except UserException as exc:
        logging.exception(exc)
        exit(1)
    except Exception as exc:
        logging.exception(exc)
        exit(2)
