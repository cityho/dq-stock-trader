import keras
from keras.models import load_model

from agent.agent2 import Agent
from functions import *
import sys

if len(sys.argv) != 4:
	print ("Usage: python evaluate.py [stock] [model] [window_size]")
	exit()

stock_name, model_name, window_size = sys.argv[1], sys.argv[2], int(sys.argv[3])
#model = load_model("models/" + model_name)
#window_size = model.layers[0].input.shape.as_list()[1]

try:
	agent = Agent(window_size, True, model_name)
except Exception as err:
	print(err)


num_df, img_fs = getStockData(stock_name, window_size)
l = len(img_fs) - 1
batch_size = 32

num_state, img_state, price = getStateV2(
		num_df, img_fs, 0, window_size + 1
	)
total_profit = 0
agent.inventory = []

for t in range(l):
	# 수정포인트 1 img, num
	action = agent.act(num_state, img_state)

	# sit => 0 다음에 t를 통해서 한칸씩 rolling 하는구조
	nxt_num_state, nxt_img_state, nxt_price = getStateV2(
		num_df, img_fs, t + 1, window_size + 1
	)
	reward = 0

	if action == 1: # buy
		agent.inventory.append(nxt_price[t])
		print ("Buy: " + formatPrice(nxt_price[t]))

	elif action == 2 and len(agent.inventory) > 0: # sell
		bought_price = agent.inventory.pop(0)
		reward = max(nxt_price[t] - bought_price, 0)
		total_profit += nxt_price[t] - bought_price
		print ("Sell: " + formatPrice(nxt_price[t]) + " | Profit: " + formatPrice(nxt_price[t] - bought_price))

	done = True if t == l - 1 else False
	#agent.memory.append((state, action, reward, next_state, done))
	num_state, img_state = nxt_num_state, nxt_img_state

	if done:
		print ("--------------------------------")
		print (stock_name + " Total Profit: " + formatPrice(total_profit))
		print ("--------------------------------")
