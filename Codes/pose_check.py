import pickle

with open('pose_data.pkl','rb') as f:
    data=pickle.load(f)

data=list(set(data['labels']))

print(f'There are {len(data)} Poses currently:')
for i in data:
    print(f' -{i}')