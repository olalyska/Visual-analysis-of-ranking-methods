import time
# from pyDecision.algorithm import vikor_method, ranking
from pyDecision.algorithm import ranking
from download_dataset import *

# COMPLETE THE DATA
step = 0.01
criteria_count = 2
threshold_list = [0]
strategy_coefficient_list = [0.5]
criterion_type_val = 'max'
W_val = [0.5]
MIDDLE_POINT = np.array([0.5 for i in range(0, criteria_count)])
####################

dataset = download_dataset(step, criteria_count)

def vikor_method(dataset, weights, criterion_type, strategy_coefficient = 0.5, graph = True, verbose = True):
    X     = np.copy(dataset)
    w     = np.copy(weights)
    best  = np.zeros(X.shape[1])
    worst = np.zeros(X.shape[1])
    for i in range(0, dataset.shape[1]):
        if (criterion_type[i] == 'max'):
            best[i]  = np.max(X[:, i])
            worst[i] = np.min(X[:, i])
        else:
            best[i]  = np.min(X[:, i])
            worst[i] = np.max(X[:, i])
    s_i = w * ( abs(best - X) / (abs(best - worst) + 0.0000000000000001) )
    r_i = np.max(s_i, axis = 1)
    s_i = np.sum(s_i, axis = 1)
    s_best = np.min(s_i)
    s_worst = np.max(s_i)
    r_best = np.min(r_i)
    r_worst = np.max(r_i)
    # fix the method - in some cases division by zero occurs
    if s_worst - s_best == 0 or r_worst - r_best == 0:
        q_i = strategy_coefficient * ((s_i - s_best) / 0.5) + (1 - strategy_coefficient) * (
                (r_i - r_best) / 0.5)
    else:
        q_i = strategy_coefficient * ((s_i - s_best) / (s_worst - s_best)) + (1 - strategy_coefficient) * (
                    (r_i - r_best) / (r_worst - r_best))

    dq = 1 /(X.shape[0] - 1)
    flow_s = np.copy(s_i)
    flow_s = np.reshape(flow_s, (s_i.shape[0], 1))
    flow_s = np.insert(flow_s, 0, list(range(1, s_i.shape[0]+1)), axis = 1)
    flow_s = flow_s[np.argsort(flow_s[:, 1])]
    flow_r = np.copy(r_i)
    flow_r = np.reshape(flow_r, (r_i.shape[0], 1))
    flow_r = np.insert(flow_r, 0, list(range(1, r_i.shape[0]+1)), axis = 1)
    flow_r = flow_r[np.argsort(flow_r[:, 1])]
    flow_q = np.copy(q_i)
    flow_q = np.reshape(flow_q, (q_i.shape[0], 1))
    flow_q = np.insert(flow_q, 0, list(range(1, q_i.shape[0]+1)), axis = 1)
    flow_q = flow_q[np.argsort(flow_q[:, 1])]
    condition_1 = False
    condition_2 = False
    if (flow_q[1, 1] - flow_q[0, 1] >= dq):
        condition_1 = True
    if (flow_q[0,0] == flow_s[0,0] or flow_q[0,0] == flow_r[0,0]):
        condition_2 = True
    solution = np.copy(flow_q)
    if (condition_1 == True and condition_2 == False):
        solution = np.copy(flow_q[0:2,:])
    elif (condition_1 == False and condition_2 == True):
        for i in range(solution.shape[0] -1, -1, -1):
            if(solution[i, 1] - solution[0, 1] >= dq):
              solution = np.delete(solution, i, axis = 0)
    if (verbose == True):
        for i in range(0, solution.shape[0]):
            print('a' + str(i+1) + ': ' + str(round(solution[i, 0], 2)))
    if ( graph == True):
        ranking(solution)
    return flow_s, flow_r, flow_q, solution

def create_VIKOR_dataset(threshold, dataset, strategy_coefficient=0.5, W_val = W_val, criterion_type_val=criterion_type_val):
    W = np.full((1, criteria_count), W_val)[0]
    criterion_type = np.full((1, criteria_count), criterion_type_val)[0]

    global final_scores_matrix
    final_scores_matrix = np.array([])
    i=0
    for row in dataset:

        matrix = np.vstack((row, MIDDLE_POINT))
        s, r, q, c_solution = vikor_method(matrix, W, criterion_type, strategy_coefficient, graph=False, verbose=False)

        c_solution = c_solution[c_solution[:, 0].argsort()]
        point_closeness = c_solution[0][1]
        middle_point_closeness = c_solution[1][1]

        # translate the result to numbers(so they can correspond in visualisation with colors)
        if abs(middle_point_closeness - point_closeness) <= threshold:
            final_scores_matrix = np.append(final_scores_matrix, 2)
        elif point_closeness > middle_point_closeness:
            final_scores_matrix = np.append(final_scores_matrix, 3)
        elif point_closeness < middle_point_closeness:
            final_scores_matrix = np.append(final_scores_matrix, 1)
        else:
            print("error")

    final_scores_matrix = final_scores_matrix.reshape(-1, 1)
    df = join_scores_matrix_and_dataset(final_scores_matrix, dataset)
    save_data_to_file(df=df, filename=f"data/VIKOR_2x/VIK_2x_V{threshold}.csv")

full_time = 0
counter = 0
for threshold in threshold_list:
    for st_coef in strategy_coefficient_list:
        start_time = time.perf_counter()
        create_VIKOR_dataset(threshold, dataset, st_coef)
        counter += 1
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        full_time += elapsed_time
        print(f"Elapsed time: {elapsed_time} seconds")

average_time = full_time / counter
print(f"Average time: {average_time} seconds")


