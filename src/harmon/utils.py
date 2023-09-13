import os
import time
from typing import Iterable

POSITIVE_ANSWERS = ["y", "yes", "yep", "yeah"]
NEGATIVE_ANSWERS = ["n", "no", "nope", "nah"]


def ask_yes_no_question(prompt: str, default_answer="y") -> bool:
    default_answer = default_answer.lower().strip()

    default_indicator = (
        "Y/n"
        if default_answer in POSITIVE_ANSWERS
        else "y/N"
        if default_answer in NEGATIVE_ANSWERS
        else "y/n"
    )

    answer = input(f"{prompt} [{default_indicator}]: ").lower().strip()

    while True:
        answer = answer or default_answer

        if answer in POSITIVE_ANSWERS:
            return True

        if answer in NEGATIVE_ANSWERS:
            return False

        answer = input(f"Invalid answer: {answer!r}. {prompt} [Y/n]: ").lower()


def select_from(options: Iterable, default=None) -> str:
    options = list(options)

    if not options:
        raise ValueError("Options list cannot be empty")

    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")

    has_default = default in options
    default_indicator = f" [default={default!r}]" if has_default else ""

    while True:
        prompt = f"Select an option (1-{len(options)}){default_indicator}: "
        choice = input(prompt).strip() or default

        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(options):
                return options[index]

        if choice in options:
            return choice

        print(f"Invalid choice: {choice}. Please select a valid option.")


def subdirectories(folder_path: str):
    for subdir in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, subdir)):
            yield subdir


def latest_modified_subdirectory(directory):
    # Get a list of all subdirectories in the specified directory
    subdirectories = (
        subdir
        for subdir in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, subdir))
    )

    # Find the latest modified subdirectory
    latest_subdirectory = max(
        subdirectories,
        key=lambda subdir: os.path.getmtime(os.path.join(directory, subdir)),
        default=None,
    )

    return latest_subdirectory
