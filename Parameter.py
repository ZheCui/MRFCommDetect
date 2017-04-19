# Dataset
# dataset = "DBLP"
# dataset = "coauthorship"
# dataset = 'synthetic'
# dataset = 'amazon'
dataset = 'youtube'

# zipf distribution
# constant cost
theta = 0
consider_cost = False
c = 0
# theta = 0.8
# consider_cost = True
# c = 1

# budget
# DBLP budget list
# budget_list = [115, 150, 200, 300, 400, 500, 700, 1000, 1500, 2000]
# coauthorship budget list
# budget_list = [1000, 1050, 1100, 1200, 1300, 1500, 1600, 1800, 2000, 2200]
# synthetic data budget list, cluster: 5, nodes: 739
# budget_list = [740, 800, 850, 900, 1000, 1100, 1200, 1500, 1800, 2000]
# amazon data budget list, cluster: 20, nodes: 400
# budget_list = [400, 450, 500, 600, 700, 800, 900, 1000, 1200, 1500]
# youtube data budget list, cluster: 20, nodes: 600
budget_list = [600, 650, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]

# sample strategy
sample_mode = 7
# runs of algorithm
test_num = 1
