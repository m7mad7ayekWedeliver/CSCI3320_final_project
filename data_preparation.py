# import pandas as pd
#
# # Load the data set
# df = pd.read_csv('health_data.csv')
# print(df.isnull().sum())
# df = df.dropna()
# print(df.duplicated().sum())
# df = df.drop_duplicates()
# df['PhysActivity_squared'] = df['PhysActivity']**2
# df = df[df['PhysActivity'] < 50]
# df.plot(x='PhysActivity', y='Smoker', kind='scatter')
#
import pandas as pd

# Load the data set
df = pd.read_csv('health_data.csv')
print(df.isnull().sum())
df = df.dropna()
print(df.duplicated().sum())
df = df.drop_duplicates()
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['age_squared'] = df['Age']**2
df['CholCheck_log'] = pd.np.log(df['CholCheck'])
df['Phys'] = df['PhysHlth'] / 2.205
def convert_to_text(x):
    if x < 6:
        return 'babe'
    elif x < 12:
        return 'boy'
    else:
        return 'man'
df['Age_tex'] = df['Age'].apply(convert_to_text)
df = df[df['age'] < 50]
df.plot(x='age', y='bp_log', kind='scatter')