import time
from pyDecision.algorithm import promethee_i
from download_dataset import *

# COMPLETE THE DATA
step = 0.01
criteria_count = 2
Q_list = [0.1]
S_list = [0]  # (-1,1)
P_list = [0.2]
F_list = ['t4'] # 't1' = Usual; 't2' = U-Shape; 't3' = V-Shape; 't4' = Level; 't5' = V-Shape with Indifference; 't6' = Gaussian; 't7' = C-Form
W_list = [0.5]
MIDDLE_POINT = np.array([[0.5 for i in range(0, criteria_count)]], dtype=float)
####################

# Download set of points
dataset = download_dataset_with_middle_point(step, criteria_count)

def create_pro_I_dataset(Q_val, S_val, P_val, W_val, F_val):
    # Initiate variables
    Q = np.full((1, criteria_count), Q_val)[0]
    S = np.full((1, criteria_count), S_val)[0]
    P = np.full((1, criteria_count), P_val)[0]
    W = np.full((1, criteria_count), W_val)[0]
    F = np.full((1, criteria_count), F_val)[0]

    global final_scores_matrix
    final_scores_matrix = np.array([])
    for row in dataset:
        matrix = np.vstack((row, MIDDLE_POINT))
        p1 = promethee_i(matrix, W=W, Q=Q, S=S, P=P, F=F, graph=False)
        score = p1[0][1]

        # translate the result to numbers(so they can correspond in visualisation with colors)
        if score == 'P+':
            final_scores_matrix = np.append(final_scores_matrix, 1)
        elif score == 'I':
            final_scores_matrix = np.append(final_scores_matrix, 2)
        elif score == '-':
            final_scores_matrix = np.append(final_scores_matrix, 3)
        elif score == 'R':
            final_scores_matrix = np.append(final_scores_matrix, 0)
        final_scores_matrix = final_scores_matrix.reshape(-1, 1)

    df = join_scores_matrix_and_dataset(final_scores_matrix, dataset)

    # save the data to file
    save_data_to_file(df, f"data/PROMETHEE_I_2x_step{step}/PRO_I_2x_Q{Q_val}_S{S_val}_P{P_val}_F({F_val}).csv")


full_time = 0
counter = 0
for Q in Q_list:
    for S in S_list:
        for P in P_list:
            for F in F_list:
                for W in W_list:
                    start_time = time.perf_counter()
                    create_pro_I_dataset(Q_val=Q, S_val=S, P_val=P, W_val=W, F_val=F)
                    counter += 1
                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time
                    full_time += elapsed_time
                    print(f"Elapsed time: {elapsed_time} seconds")

average_time = full_time/counter
print(f"Average time: {average_time} seconds")