import pandas as pd

df = pd.DataFrame({'num_legs': [2.02, 4.12], 'num_wings': [2, 0]},
                  index=['falcon', 'dog'])

print(df.isin([0, 2]))