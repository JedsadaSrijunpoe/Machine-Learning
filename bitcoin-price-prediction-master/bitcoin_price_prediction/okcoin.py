"""Script to gather market data from OKCoin Spot Price API."""
import requests
from pytz import utc
from datetime import datetime
from pymongo import MongoClient
from apscheduler.schedulers.blocking import BlockingScheduler

client = MongoClient()
database = client['okcoindb']
collection = database['historical_data2']


def tick():
    """Gather market data from OKCoin Spot Price API and insert them into a
       MongoDB collection."""
    ticker = requests.get('https://www.okcoin.com/api/v5/market/ticker?instId=BTC-USD').json()
    depth = requests.get('https://www.okcoin.com/api/v5/market/books?instId=BTC-USD&sz=60').json()
    print(ticker["data"][0]['ts'])
    date = datetime.fromtimestamp(int(ticker["data"][0]['ts']) / 1000)
    price = float(ticker['data'][0]['last'])
    v_bid = sum([float(bid[1]) for bid in depth["data"][0]['bids']])
    v_ask = sum([float(ask[1]) for ask in depth["data"][0]['asks']])
    collection.insert_one({'date': date, 'price': price, 'v_bid': v_bid, 'v_ask': v_ask})
    print(date, price, v_bid, v_ask)


def main():
    """Run tick() at the interval of every ten seconds."""
    scheduler = BlockingScheduler(timezone=utc)
    scheduler.add_job(tick, 'interval', seconds=10)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == '__main__':
    main()
