import time
from pyDecision.algorithm import topsis_method
from download_dataset import *


# COMPLETE THE DATA
step = 0.01
criteria_count = 2
threshold_list = [0]
W_val = 0.5  # Weights Value
criterion_type_val = 'max' # Criterion Type: 'max' or 'min'
MIDDLE_POINT = np.array([[0.5 for i in range(0, criteria_count)]], dtype=float)
####################


# Download set of points
dataset = download_dataset_with_middle_point(step, criteria_count)

def create_topsis_dataset(threshold, dataset, W_val = W_val, criterion_type_val=criterion_type_val):
    weights = np.full((1, criteria_count), W_val)[0]
    criterion_type = np.full((1, criteria_count), criterion_type_val)[0]
    closeness_matrix = topsis_method(dataset, weights, criterion_type, graph=False, verbose=False)
    middle_point_closeness = closeness_matrix[-1]

    # translate the result to numbers(so they can correspond in visualisation with colors)
    final_scores_matrix = np.array([])
    for i in range(0, len(closeness_matrix)):
        point_closeness = closeness_matrix[i]
        if abs(middle_point_closeness - point_closeness) <= threshold:
            final_scores_matrix = np.append(final_scores_matrix, 2)
        elif point_closeness > middle_point_closeness:
            final_scores_matrix = np.append(final_scores_matrix, 1)
        elif point_closeness < middle_point_closeness:
            final_scores_matrix = np.append(final_scores_matrix, 3)
        else:
            print("error")
        final_scores_matrix = final_scores_matrix.reshape(-1, 1)

    df = join_scores_matrix_and_dataset(final_scores_matrix, dataset)
    save_data_to_file(df=df, filename=f"data/TOPSIS_1000x/TOP_1000x_TH{threshold}.csv")

full_time = 0
counter = 0
for threshold in threshold_list:
    start_time = time.perf_counter()
    create_topsis_dataset(threshold, dataset)
    counter += 1
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    full_time += elapsed_time
    print(f"Elapsed time: {elapsed_time} seconds")

average_time = full_time/counter
print(f"Average time: {average_time} seconds")

