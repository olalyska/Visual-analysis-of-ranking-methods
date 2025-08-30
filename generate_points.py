import itertools
from download_dataset import *

# ADJUST THE DATA HERE #
step = 0.25
criteria_count = 2  # number of criteria
########################


def create_dataset(step, criteria_count, filename):
    values = np.arange(0, 1 + step, step)
    dataset = np.array(list
            (itertools.product(values, repeat=criteria_count)))
    df = pd.DataFrame(dataset)
    df = df.round(decimals=4)
    save_data_to_file(df, filename)


create_dataset(step, criteria_count,
         filename=f"data/datasets/dataset_s{step}_c{criteria_count}.csv")
