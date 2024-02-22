import pandas as pd 
import argparse
import os 
from deephive.environment.utils import parse_config

config_path = "config/exp_config.json"
# load config
config = parse_config(config_path)

parser = argparse.ArgumentParser()  

parser.add_argument("--n_dim", type=int, default=2)

# parser.add_argument("--exp_list", type=str, default="2,3")

args = parser.parse_args()

result_path = f"experiments/results_{args.n_dim}/"

exp_list = [i for i in range(0, 29)]

dfs = []
for exp in exp_list:
    try:
        exp_path = os.path.join(result_path, f"result_comparison_function_{exp}.csv")
        df = pd.read_csv(exp_path)
        # keep only the first 2 and rhe last row of the dataframe
        df = df.iloc[[0, 1, -1]]
        # do  not add empty dataframes
        if not df.empty:
            dfs.append(df)
        else:
            print(f"File {exp_path} is empty")
    except:
        print(f"File {exp_path} not found")
        continue
    

def create_benchmark_tables(dfs):
    # Step 1: Extract unique algorithm names and functions
    # Assuming algorithm names and functions can be extracted from the 'title' column
    algos = set()
    functions = set()
    for df in dfs:
        df['Function'], df['Algorithm'] = zip(*df['title'].apply(lambda x: x.split(' - ', 1)))
        algos.update(df['Algorithm'].unique())
        functions.update(df['Function'].unique())

    # Step 2: Create the MultiIndex for columns
    columns = pd.MultiIndex.from_product([sorted(algos), ['mean', 'mid_mean']], names=['Algorithm', 'Metric'])

    # Initialize an empty DataFrame to hold the aggregated data
    combined_df = pd.DataFrame(columns=columns, index=sorted(functions))

    # Step 3: Aggregate data
    for df in dfs:
        for index, row in df.iterrows():
            func = row['Function']
            algo = row['Algorithm']
            combined_df.loc[func, (algo, 'mean')] = row['mean']
            combined_df.loc[func, (algo, 'mid_mean')] = row['mid_mean']

    # Reset index name
    combined_df.index.name = 'Function'
    
    def highlight_top3(s):
        """
        Highlight the first, second, and third minimum in a Series.
        """
        # Ensure numeric values and drop NaNs for accurate comparison
        numeric_s = pd.to_numeric(s, errors='coerce').dropna()
        if numeric_s.empty:
            return ['' for _ in s]
        min_vals = numeric_s.nsmallest(3)
        # Use different colors for first, second, and third minimum
        colors = {min_vals.index[0]: 'background-color: green',
                min_vals.index[1]: 'background-color: red'
                #min_vals.index[2]: 'background-color: blue'
                }
        return [colors.get(i, '') for i in s.index]

    def highlight_min(s):
        '''
        highlight the maximum in a Series yellow.
        '''
        is_min = s == s.min()
        return ['background-color: green' if v else '' for v in is_min]

    # Step 4: Display the combined DataFrame
    return combined_df.style.apply(highlight_top3, axis=1).apply(highlight_min, axis=1)


try:
    result_comparison = create_benchmark_tables(dfs)
    # save the result_comparison
    result_comparison.to_excel(f"{result_path}result_comparison_function_no_split.xlsx")
except Exception as e:
    print("Error creating the benchmark tables")
    print(e)
    import traceback; traceback.print_exc();