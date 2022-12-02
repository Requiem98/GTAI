from libraries import *
import baseFunctions as bf


data = pd.read_csv(DATA_ROOT_DIR + 'data.csv', index_col=0)

for i in range(21):
    i+1

i


o = normalize_steering(data['steeringAngle'][(data['steeringAngle'] == 0)].to_numpy(dtype=np.float32))


bf.reverse_normalized_steering(o)

