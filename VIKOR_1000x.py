import time
from pyDecision.algorithm import vikor_method, ranking
from download_dataset import *

# COMPLETE THE DATA
step = 0.01
criteria_count = 2
threshold_list = [0]
criterion_type_val = 'max'
W_val = [0.5]
MIDDLE_POINT = np.array([0.5 for i in range(0, criteria_count)])
####################

dataset = download_dataset_with_middle_point(step, criteria_count)


def create_VIKOR_dataset(threshold, dataset, W_val = W_val, criterion_type_val=criterion_type_val):
    W = np.full((1, criteria_count), W_val)[0]
    criterion_type = np.full((1, criteria_count), criterion_type_val)[0]

    s, r, q, c_solution = vikor_method(dataset, W, criterion_type, strategy_coefficient=0.5, graph=False, verbose=False)
    c_solution = c_solution[c_solution[:, 0].argsort()]
    middle_point_closeness = c_solution[-1][1]

    # translate the result to numbers(so they can correspond in visualisation with colors)
    final_scores_matrix = np.array([])
    for i in range(0, len(c_solution)-1):
        point_closeness = c_solution[i][1]
        if abs(middle_point_closeness - point_closeness) <= threshold:
            final_scores_matrix = np.append(final_scores_matrix, 2)
        elif point_closeness > middle_point_closeness:
            final_scores_matrix = np.append(final_scores_matrix, 3)
        elif point_closeness < middle_point_closeness:
            final_scores_matrix = np.append(final_scores_matrix, 1)
        else:
            print("error")
    final_scores_matrix = final_scores_matrix.reshape(-1, 1)

    df = join_scores_matrix_and_dataset(final_scores_matrix, dataset[:-1, :])
    save_data_to_file(df=df, filename=f"data/VIKOR_1000x/VIK_1000x_TH{threshold}.csv")


full_time = 0
counter = 0
for threshold in threshold_list:
    start_time = time.perf_counter()
    create_VIKOR_dataset(threshold, dataset)
    counter += 1
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    full_time += elapsed_time
    print(f"Elapsed time: {elapsed_time} seconds")

average_time = full_time / counter
print(f"Average time: {average_time} seconds")
