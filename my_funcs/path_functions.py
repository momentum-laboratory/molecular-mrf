import re
import os

def find_n(txt_fn, target_str):
    """
    Find export file number by protocol name and txt file location
    :param txt_fn: the descriptory txt file path, root->scans->date->subject->txt file
    :param target_str: the required protocol name
    :return: n: the required export file number
    """
    # find export file num
    with open(txt_fn, 'r') as file:
        for line in file:
            match = re.search(rf'{target_str}.*\(E(\d+)\)', line)
            if match:
                n = int(match.group(1))
                return n
    raise ValueError(f"Error: '{target_str}' not found in the file '{txt_fn}'.")


def make_folder(folder_fn):
    """
    Create folder (if it doesn't exist yet)
    :param folder_fn: the folder path
    """
    try:
        # Try to create the folder
        os.makedirs(folder_fn)
        print(f"Folder '{folder_fn}' created successfully.")
    except FileExistsError:
        print(f"Folder '{folder_fn}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")