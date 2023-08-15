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
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    """
    This will send error to GPT for fixing.
    And will return a response inform of JSON file
    """
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
        f"{error_messages}\n"
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


def apply_changes(
        file_path: str, changes: List, confirm: bool = False
):
    """
    This will read and confirm/apply the changes
    """
    with open(file_path) as f:
        original_file_lines = f.readlines()

    # For finding explatation elements
    operation_changes = [change for change in changes if "operation" in change]
    explanation = [
        change["explanation"] for change in changes if "explanation" in change
    ]

    # For printing changes in reverse order so,it is seen properly on output window
    operation_changes.sort(key=lambda x: x["line"], reverse=True)

    file_lines = original_file_lines.copy()
    for change in operation_changes:
        operation = change["operation"]
        line = change["line"]
        content = change["content"]

        if operation == "Replace":
            file_lines[line - 1] = content + "\n"
        elif operation == "Delete":
            del file_lines[line - 1]
        elif operation == "InsertAfter":
            file_lines.insert(line, content + "\n")

    # For printing explanation
    cprint("Explanation: ", "blue")
    for explanation in explanations:
        cprint(f"- {explanation}", "blue")

    # For Displaying changes in diff view
    print("\nChanges to be made: ")
    diff = difflib.unified_diff(original_file_lines, file_lines, lineterm="")
    for line in diff:
        if line.startswith("+"):
            cprint(line, "green", end="")
        elif line.startswith("-"):
            cprint(line, "red", end="")
        else:
            print(line, end="")

    # For confirming through user that if he wants changes or not
    if confirm:
        confirmation = input("Do you want to apply this changes? (y/n): ")
        if confirmation.lower() != "y":
            print("Changes not applied.")
            sys.exit(0)

    # For applying changes after confirmation(y) from the user
    with open(file_path, "w") as f:
        f.writelines(file_lines)
    cprint("----------------  Changes Applied  ------------------", "green")


def checking_availability(model):
    """
    This function will check,
    if GPTs models are available,
    there token limit, etc.
    """
    models_available = [x["id"] for x in openai.Model.list()["data"]]
    if model not in models_available:
        cprint(
            f"Model {model} is not available.Try rerunning with "
            f"`{'--', models_available, '--'}` instead."
            "You can also configure a default model in .env"
        )
        exit()


def main(script_name, *script_args, revert=False, model=default_gpt_model, confirm=True):
    """
    main function for running
    """

    if revert:
        backup_file = script_name + ".bak"
        if os.path.exists(backup_file):
            shutil.copy(backup_file, script_name)
            print(f"Reverted changes to {script_name}")
            sys.exit(0)
        else:
            print(f"No BackUp file found for {script_name}")
            sys.exit(1)

    # checking model availability
    checking_availability(model)

    # For Making a BackUp file of original file
    shutil.copy(script_name, script_name + ".bak")

    while True:
        output, returncode = run_script(script_name, script_args)

        if returncode == 0:
            cprint("Script ran Successfully!!", "blue")
            print("Output: ", output)
            break
        else:
            cprint("Script Crashed.Trying to fix...", "blue")
            print("Output: ", output)
            json_response = send_error_to_gpt(
                file_path=script_name,
                args=script_args,
                error_messages=output,
                model=model,
            )

            apply_changes(script_name, json_response, confirm=confirm)
            cprint("--------  Changes Applied.Rerunning...", "blue")
