from libraries import *
import baseFunctions as bf

data = pd.read_csv(DATA_ROOT_DIR + 'data.csv', index_col=0)


def normalize_steering(x):
    x = x+40
    return x / 80


def reverse_normalized_steering(x):
    x = x*80
    return x - 40


norm_data = normalize_steering(data['steeringAngle'].to_numpy())

plt.hist(norm_data, bins=40)


inv_norm_data = reverse_normalized_steering(norm_data)


plt.hist(inv_norm_data, bins=40)
plt.hist(data['steeringAngle'].to_numpy(), bins=40, color='red')

