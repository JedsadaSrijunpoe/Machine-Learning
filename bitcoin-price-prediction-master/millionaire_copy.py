from pymongo import MongoClient
from bitcoin_price_prediction.bayesian_regression import *
import matplotlib.pyplot as plt

client = MongoClient()
database = client['okcoindb']
collection = database['historical_data']

# Retrieve price, v_ask, and v_bid data points from the database.
prices = []
v_ask = []
v_bid = []
num_points = 2800
for doc in collection.find().limit(num_points):
    prices.append(doc['price'])
    v_ask.append(doc['v_ask'])
    v_bid.append(doc['v_bid'])

# Divide prices into three, roughly equal sized, periods:
# prices1, prices2, and prices3.
[prices1, prices2, prices3] = np.array_split(prices, 3)

# Divide v_bid into three, roughly equal sized, periods:
# v_bid1, v_bid2, and v_bid3.
[v_bid1, v_bid2, v_bid3] = np.array_split(v_bid, 3)

# Divide v_ask into three, roughly equal sized, periods:
# v_ask1, v_ask2, and v_ask3.
[v_ask1, v_ask2, v_ask3] = np.array_split(v_ask, 3)

# Use the first time period (prices1) to generate all possible time series of
# appropriate length (180, 360, and 720).
timeseries180 = generate_timeseries(prices1, 180)
timeseries360 = generate_timeseries(prices1, 360)
timeseries720 = generate_timeseries(prices1, 720)

# Cluster timeseries180 in 100 clusters using k-means, return the cluster
# centers (centers180), and choose the 20 most effective centers (s1).
centers180 = find_cluster_centers(timeseries180, 100)
s1 = choose_effective_centers(centers180, 20)

centers360 = find_cluster_centers(timeseries360, 100)
s2 = choose_effective_centers(centers360, 20)

centers720 = find_cluster_centers(timeseries720, 100)
s3 = choose_effective_centers(centers720, 20)

all_dp1 = []
for i in range(720, len(prices2) - 1):
    dp = prices2[i + 1] - prices2[i]
    dp1 = predict_dpi(prices2[i - 180:i], s1)
    dp2 = predict_dpi(prices2[i - 360:i], s2)
    dp3 = predict_dpi(prices2[i - 720:i], s3)
    all_dp1.append(dp2)
    r = (v_bid2[i] - v_ask2[i]) / (v_bid2[i] + v_ask2[i])
    
plt.plot(np.array(range(len(all_dp1))), np.array(all_dp1))
plt.show()


# Use the second time period to generate the independent and dependent
# variables in the linear regression model:
# Δp = w0 + w1 * Δp1 + w2 * Δp2 + w3 * Δp3 + w4 * r.
Dpi_r, Dp = linear_regression_vars(prices2, v_bid2, v_ask2, s1, s2, s3)

# Find the parameter values w (w0, w1, w2, w3, w4).
w = find_parameters_w(Dpi_r, Dp)

# Predict average price changes over the third time period.
dps = predict_dps(prices3, v_bid3, v_ask3, s1, s2, s3, w)

# What's your 'Fuck You Money' number?
bank_balance = evaluate_performance(prices3, dps, t=0.0001, step=1)

print(bank_balance)
