from pymongo import MongoClient
from bitcoin_price_prediction.bayesian_regression import *
import matplotlib.pyplot as plt

client = MongoClient()
database = client['okcoindb']
collection = database['historical_data2']

# Retrieve price, v_ask, and v_bid data points from the database.
prices = []
v_ask = []
v_bid = []
count = -1
num_points = 9000
for doc in collection.find().limit(num_points):
    count += 1
    if count == -1+1:
        prices.append(doc['price'])
        v_ask.append(doc['v_ask'])
        v_bid.append(doc['v_bid'])
        count = -1
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

# Use the second time period to generate the independent and dependent
# variables in the linear regression model:
# Δp = w0 + w1 * Δp1 + w2 * Δp2 + w3 * Δp3 + w4 * r.
Dpi_r, Dp = linear_regression_vars(prices2, v_bid2, v_ask2, s1, s2, s3)

# Find the parameter values w (w0, w1, w2, w3, w4).
w = find_parameters_w(Dpi_r, Dp)

# Predict average price changes over the third time period.
dps = predict_dps(prices3, v_bid3, v_ask3, s1, s2, s3, w)

experiment3 = evaluate_performance_with_htime(prices3, dps, t=0.10, step=1)
bank_balance_series = experiment3["cul_p"]
prices_series = experiment3["prices_series"]
print(experiment3["total_pnl"])

total_pnl = []
pnl = []
holding_time = []
trades= []
threshold = range(1,201,10)
for i in threshold:
    result = evaluate_performance_with_htime(prices3, dps, t=i*0.001, step=1)
    total_pnl.append(result["total_pnl"])
    pnl.append(result["pnl"])
    holding_time.append(result["avg_hold_time"])
    trades.append(result["trades"])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# Plotting the first set of data
ax1_twin = ax1.twinx()
ax1.scatter(np.array(threshold)*0.001, np.array(holding_time), c='blue', label='Holding Time')
ax1_twin.scatter(np.array(threshold)*0.001, np.array(trades), c='black', label='Trades')

# Plotting the second set of data
ax2_twin = ax2.twinx()
ax2.scatter(np.array(threshold)*0.001, np.array(pnl), c='blue', label='P&L')
ax2_twin.scatter(np.array(threshold)*0.001, np.array(total_pnl), c='black', label='Total P&L')

# Setting labels and legends for the first subplot
ax1.set_ylabel('Holding Time', color='blue')
ax1_twin.set_ylabel('Trades', color='black')

# Setting labels and legends for the second subplot
ax2.set_xlabel('Threshold')
ax2.set_ylabel('P&L', color='blue')
ax2_twin.set_ylabel('Total P&L', color='black')

# Adjust layout for better visibility
plt.tight_layout()

plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(np.array(pnl), np.array(trades))
ax.set_xlabel("P&L")
ax.set_ylabel("Trades")
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
ax_twin = ax.twinx()
ax.plot(np.array(range(len(prices_series))), prices_series, c="blue")
ax_twin.plot(np.array(range(len(prices_series))), bank_balance_series, c="black")
ax.set_xlabel('Time Unit(10s)')
ax.set_ylabel('Bitcoin Price', color='blue')
ax_twin.set_ylabel('Culmulative profit', color='black')
plt.show()