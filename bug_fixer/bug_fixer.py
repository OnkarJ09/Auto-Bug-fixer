from bug_fixer.config import OPENAI_API_KEY
from typing import List, Dict
from termcolor import cprint
from dotenv import load_dotenv
import difflib
import openai
import subprocess
import shutil
import json
import os
import sys


# SetUp OpenAI API key
load_dotenv()
openai.api_key = os.getenv(OPENAI_API_KEY)

# Default Model from Openai
default_gpt_model = os.environ.get('DEFAULT_MODEL', 'gpt-3.5-turbo-16k')

# NP retries for json_validate_response
validate_json_retry = int(os.getenv('VALIDATE_JSON_RETRY', -1))       # here -1 indicates infinite tries

# for reading system prompts
with open(os.path.join(os.path.dirname(__file__), '...', 'prompt.txt'), "r") as f:
    SYSTEM_PROMPT = f.read()


def run_script(script_name: str, script_args: List) -> str:
    """
    If Scripts name end with '.py' then this func will run it in python
    and if not
    then it will run it in node
    """
    script_args = [str(arg) for arg in script_args]
    subprocess_args = (
        [sys.executable, script_name, *script_args]
        if script_name.endswith(".py")
        else ["node", script_name, *script_args]
    )

    try:
        result = subprocess.check_output(subprocess_args, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as error:
        return error.output.decode("utf-8"), error.returncode
    return result.decode("utf-8"), 0


def json_validate_response(
        model: str, messages: List[Dict], np_retry: int = validate_json_retry
) -> Dict:
    """
    This func will convert non-json to json responses.
    This will run validate_json_retry continuously,
    as validate_json_retry has value set to -1 it will run in an Infinite Loop
    until the responses grnrrated is json response.
    """
    json_response = {}
    if np_retry != 0:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.5
        )
        messages.append(response.choices[0].message)
        content = response.choices[0].message.content
        # to see if json can be resolved
        try:
            json_start_index = content.index(
                "["       # to find staring position of json data
            )
            json_data = content[
                json_start_index     # to extract data from the string
            ]
            json_response = json.loads(json_data)
            return json_response
        except (json.decoder.JSONDecodeError, ValueError) as e:
            cprint(f"{e}. Trying Rerunning the query,", "red")
            # to debug
            cprint(f"\nGPT Response:\n\n{content}\n\n", "yellow")
            # This will write a user message that will tell json is invalid
            messages.append(
                {
                    "role": 'user',
                    "content": (
                        "Your response could not be resolved by json.loads."
                        "PLEASE, Rewrite your last message as pure json."
                    ),
                }
            )
            # For decreasing np_retry
            np_retry -= 1
            # For reruning API-call
            return json_validate_response(model, messages, np_retry)
        except Exception as e:
            cprint(f"Unknwn error: {e}", "red")
            cprint(f"\nGPT Response:\n\n{content}\n\n", "yellow")
            raise e
    raise Exception(
        f"No valid json response found after {validate_json_retry} tries. Now Exiting."
    )


def send_error_to_gpt(
        file_path: str, args: List, error_messages: str, model: str = default_gpt_model
) -> Dict:
    with open(file_path, "r") as f:
        file_line = f.readlines()

    file_with_lines = []
    for i, line in enumerate(file_line):
        file_with_lines.append(str(i + 1) + ":" + line)
    file_with_lines = "".join(file_with_lines)

    prompt = (
        "Here is the script that needs fixing:\n\n"
        f"{file_with_lines}\n\n"
        "Here are the arguments it has provided:\n\n"
        f"{args}\n\n"
        "Here is the error message:\n\n"
        f"{error_message}\n"
        "Please provide your suggested changes, and remember to stick to the"
        "exact format as described above."
    )

    # For printing prompts
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    return json_validate_response(model, messages)


