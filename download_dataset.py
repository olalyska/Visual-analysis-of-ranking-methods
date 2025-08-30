from numpy import genfromtxt
import numpy as np
import pandas as pd
import os

def download_dataset_with_middle_point(step, criteria_count, M_val=0.5):
    MIDDLE_POINT = np.array([[M_val for i in range(0, criteria_count)]], dtype=float)
    try:
        data = genfromtxt(f'data/datasets/dataset_s{step}_c{criteria_count}.csv', delimiter=',')
    except:
        print(
            f"Generate data via generete_points.py script with step = {step} and criteria_count = {criteria_count}")
        exit(1)
    dataset = np.concatenate((data, MIDDLE_POINT), axis=0)
    return dataset


def download_dataset(step, criteria_count, M_val=0.5):
    try:
        data = genfromtxt(f'data/datasets/dataset_s{step}_c{criteria_count}.csv', delimiter=',')
    except:
        print(
            f"Generate data via generete_points.py script with step = {step} and criteria_count = {criteria_count}")
        exit(1)
    return data


def join_scores_matrix_and_dataset(final_scores_matrix, dataset):
    final_scores_matrix = final_scores_matrix.astype(np.int64)
    points_plus_final_scores = np.hstack((dataset, final_scores_matrix))
    points_plus_final_scores = points_plus_final_scores.astype(object)
    points_plus_final_scores[:, -1] = points_plus_final_scores[:, -1].astype(int)
    # df = pd.DataFrame(points_plus_final_scores[:-1])
    df = pd.DataFrame(points_plus_final_scores[:])
    return df


def save_data_to_file(df, filename):
    try:
        df.to_csv(filename, index=False, header=False)
    except:
        file_parts = filename.split("/")
        dir_list = [file_parts[0], file_parts[1]]
        directory = '/'.join(dir_list)
        os.mkdir(directory)
        print(f"Directory: {directory} created")
        df.to_csv(filename, index=False, header=False)
    print(f"Dataset: {filename} created")


def get_middle_point_index(dataset, MIDDLE_POINT):
    counter = 0
    for i in dataset:
        if i[0] == MIDDLE_POINT[0][0] and i[1] == MIDDLE_POINT[0][1]:
            print(counter)
            break
        counter += 1

