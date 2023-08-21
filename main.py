
import csv
import os
import json

from timeit import default_timer as timer

from approximate_entropy import ApproximateEntropy

def calculate_and_save_apen_values_into_csv_test(path : str,
                                            save_dir : str, 
                                            use_memoization : bool = True):
    """
    *For test purposes, has the choice to use or not use memoization.
    Calculates the apen values of json drawing data stored
    in the directory given by path, and save its list as a 
    csv into the directory given by save_dir.

    :param str path: The directory in which all json drawing data
    we want to calculate the apen value for reside.
    :param str save_dir: The directory to which we save the resulting
    apen values.
    :param bool use_memoization: Whether to use memoization or not - put
    for testing. Defaults to True and should be used as true for normal use.
    """
    apen_values = {}
    all_files = os.listdir(path)
    
    start_time = timer()

    for i, file in enumerate(all_files):
        print(f"Reading file: {file} - {i+1}th file out of {len(all_files)}!")
        full_path = f"{path}/{file}"

        with open(full_path, 'r') as f:
            json_content = f.read()
            content = json.loads(json_content)
            userId = content["userId"]
            isDominant = content["dominantHand"] == content["drawnHand"]

        apen_val = ApproximateEntropy.approximate_entropy_from_file(
            file_path=full_path, use_memoization=use_memoization
            )
        apen_values[userId] = {"isDominant" : isDominant, "apen_val" : apen_val}

        print(f"Finished reading file: {file}!")
    
    end_time = timer()
    
    print(f"Execution time is: {end_time - start_time} fractional seconds!")
    
    saved_file_name = f"{path.replace('/', '_')}_all_apen_values.csv"
    with open(f"{save_dir}/{saved_file_name}", "w") as csvfile:
        fieldnames = ['userId', 'isDominant', 'apen_val']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for key in apen_values.keys():
            val = apen_values[key]
            writer.writerow({'userId': key, 'isDominant': val["isDominant"], 'apen_val': val["apen_val"]})


def calculate_and_save_apen_values_into_csv(path : str,
                                            save_dir : str):
    """
    Calculates the apen values of json drawing data stored
    in the directory given by path, and save its list as a 
    csv into the directory given by save_dir.

    :param str path: The directory in which all json drawing data
    we want to calculate the apen value for reside.
    :param str save_dir: The directory to which we save the resulting
    apen values.
    """
    calculate_and_save_apen_values_into_csv_test(path=path, 
                                                 save_dir=save_dir,
                                                 use_memoization=True)