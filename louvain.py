import numpy as np

def directed_louvain(adj):
	n_nodes = adj.shape[0]
	adj = abs(adj)
	in_deg = adj.sum(axis=0)
	out_deg = adj.sum(axis=1)
	weight_sum = adj.sum()

	comms = {
		i: {
			"id": i,
			"members": set([i]), 		# a set of node ids indicating membership 
			"out_deg_sum": out_deg[i], 	# the sum of the outdegrees of every node in this community
			"in_deg_sum": in_deg[i], 	# the sum of the indegrees of every node in this community
			"k_out": adj[i].copy(),		# the number of incoming edges each node has connected to nodes in this community
			"k_in": adj[:, i].copy()	# the number of outgoing edges each node has connected to nodes in this community
		} for i in range(n_nodes)
	}

	# keep track of which community each node belongs to
	node_map = {i: i for i in range(n_nodes)}

	modularity = 1. / weight_sum * (adj.diagonal().sum() \
				 - 1. / weight_sum * (out_deg * in_deg).sum())

	while True:

		old_modularity = modularity

		# Stage 1 (greedy search over partitions)
		while True:

			# flag to know if any communities changed on this pass
			changed = False

			for i in range(n_nodes):

				neighbors_in = set(np.where(adj[:, i] > 0)[0])
				neighbors_out = set(np.where(adj[i] > 0)[0])
				neighbors = neighbors_in.union(neighbors_out)

				best_delta = 0
				best_comm = None
				comms_tried = set([node_map[i]])

				for j in neighbors:

					# if we already tried to move node i into node j's community,
					# don't bother trying again
					if node_map[j] in comms_tried:
						continue	
					comms_tried.add(node_map[j])

					# calculate the change in modularity due to adding node i
					# to node j's community
					new_comm = comms[node_map[j]]
					delta_add = 1. / weight_sum * new_comm["k_in"][i] \
								+ 1. / weight_sum * new_comm["k_out"][i] \
								- in_deg[i] / (weight_sum ** 2) * new_comm["out_deg_sum"] \
								- out_deg[i] / (weight_sum ** 2) * new_comm["in_deg_sum"] \
								+ 1. / weight_sum * (adj[i, i] - (in_deg[i] * out_deg[i]) / weight_sum)

					# calculate the change in modularity due to removing node i
					# from its current community
					old_comm = comms[node_map[i]]
					delta_remove = -1. / weight_sum * (old_comm["k_in"][i] - adj[i, i]) \
								   - 1. / weight_sum * (old_comm["k_out"][i] - adj[i, i]) \
								   + in_deg[i] / (weight_sum ** 2) * (old_comm["out_deg_sum"] - out_deg[i]) \
								   + out_deg[i] / (weight_sum ** 2) * (old_comm["in_deg_sum"] - in_deg[i]) \
								   - 1. / weight_sum * (adj[i, i] - (in_deg[i] * out_deg[i]) / weight_sum)

					# keep track of the overall change
					delta = delta_add + delta_remove
					if delta > best_delta:
						best_delta = delta
						best_comm = new_comm

				# if we found a positive change in modularity, then move node i into its new
				# community
				if best_delta > 0:
					modularity += best_delta
					changed = True
					old_comm = comms[node_map[i]]
					best_comm["members"].add(i)
					best_comm["out_deg_sum"] += out_deg[i]
					best_comm["in_deg_sum"] += in_deg[i]
					best_comm["k_out"] += adj[i]
					best_comm["k_in"] += adj[:, i]
					node_map[i] = best_comm["id"]
					old_comm["members"].remove(i)
					if len(old_comm["members"]) == 0:
						del comms[old_comm["id"]]
					else:
						old_comm["out_deg_sum"] -= out_deg[i]
						old_comm["in_deg_sum"] -= in_deg[i]
						old_comm["k_out"] -= adj[i]
						old_comm["k_in"] -= adj[:, i]

			# if there are no more improvements to be made, go on to stage 2
			if not changed:
				break

		# Stage 2 (define a new graph whose nodes are the communities from stage 1)

		# if the modularity didn't increase during stage 1, return the local maximum	
		if old_modularity == modularity:
			return modularity

		# form the new graph and put each new node into its own community	
		n_new_nodes = len(comms)
		new_adj = np.zeros((n_new_nodes, n_new_nodes))
		new_comms = {}
		k = 0
		for i in comms:
			new_comms[k] = {
				"id": k,
				"members": set([k]),
				"out_deg_sum": comms[i]["out_deg_sum"],
				"in_deg_sum": comms[i]["in_deg_sum"],
				"k_out": np.zeros(n_new_nodes),
				"k_in": np.zeros(n_new_nodes)
			}
			k += 1
		k = 0
		for i in comms:
			l = 0
			for j in comms:
				new_adj[k, l] = comms[i]["k_out"][list(comms[j]["members"])].sum()
				new_comms[k]["k_out"][l] = new_adj[k, l]
				new_comms[l]["k_in"][k] = new_adj[k, l]
				l += 1
			k += 1
		adj = new_adj
		comms = new_comms
		in_deg = adj.sum(axis=0)
		out_deg = adj.sum(axis=1)
		n_nodes = len(comms)
		node_map = {i: i for i in range(n_nodes)}
		weight_sum = adj.sum()


def undirected_louvain(adj):
	n_nodes = adj.shape[0]
	adj = abs(adj)
	deg = adj.sum(axis=0)
	weight_sum = adj.sum() / 2.

	comms = {
		i: {
			"id": i,
			"members": set([i]), 	# a set of node ids indicating membership 
			"deg_sum": deg[i], 		# the sum of the degrees of every node in this community
			"k_in": adj[i].copy()	# the number of edges each node has connected to nodes in this community
		} for i in range(n_nodes)
	}

	# keep track of which community each node belongs to
	node_map = {i: i for i in range(n_nodes)}

	modularity = 1. / (2 * weight_sum) * (adj.diagonal().sum() \
				 - 1. / (2 * weight_sum) * (deg ** 2).sum())

	while True:

		old_modularity = modularity

		# Stage 1 (greedy search over partitions)
		while True:

			# flag to know if any communities changed on this pass
			changed = False

			for i in range(n_nodes):

				neighbors = list(np.where(adj[i] > 0)[0])

				best_delta = 0
				best_comm = None
				comms_tried = set([node_map[i]])

				for j in neighbors:

					# if we already tried to move node i into node j's community,
					# don't bother trying again
					if node_map[j] in comms_tried:
						continue	
					comms_tried.add(node_map[j])

					# calculate the change in modularity due to adding node i
					# to node j's community
					new_comm = comms[node_map[j]]
					delta_add = 1. / weight_sum * new_comm["k_in"][i] \
								- deg[i] / (2. * weight_sum ** 2) * new_comm["deg_sum"] \
								+ 1. / (2 * weight_sum) * (adj[i, i] - (deg[i] ** 2) / (2 * weight_sum))

					# calculate the change in modularity due to removing node i
					# from its current community
					old_comm = comms[node_map[i]]
					delta_remove = -1. / weight_sum * (old_comm["k_in"][i] - adj[i, i]) \
								+ deg[i] / (2 * weight_sum ** 2) * (old_comm["deg_sum"] - deg[i]) \
								- 1. / (2 * weight_sum) * (adj[i, i] - (deg[i] ** 2) / (2 * weight_sum))

					# keep track of the overall change
					delta = delta_add + delta_remove
					if delta > best_delta:
						best_delta = delta
						best_comm = new_comm

				# if we found a positive change in modularity, then move node i into its new
				# community
				if best_delta > 0:
					modularity += best_delta
					changed = True
					old_comm = comms[node_map[i]]
					best_comm["members"].add(i)
					best_comm["deg_sum"] += deg[i]
					best_comm["k_in"] += adj[i]
					node_map[i] = best_comm["id"]
					old_comm["members"].remove(i)
					if len(old_comm["members"]) == 0:
						del comms[old_comm["id"]]
					else:
						old_comm["deg_sum"] -= deg[i]
						old_comm["k_in"] -= adj[i]

			# if there are no more improvements to be made, go on to stage 2
			if not changed:
				break

		# Stage 2 (define a new graph whose nodes are the communities from stage 1)

		# if the modularity didn't increase during stage 1, return the local maximum	
		if old_modularity == modularity:
			return modularity

		# form the new graph and put each new node into its own community	
		n_new_nodes = len(comms)
		new_adj = np.zeros((n_new_nodes, n_new_nodes))
		new_comms = {}
		k = 0
		for i in comms:
			l = 0
			new_comms[k] = {
				"id": k,
				"members": set([k]),
				"deg_sum": comms[i]["deg_sum"],
				"k_in": np.zeros(n_new_nodes)
			}
			for j in comms:
				new_adj[k, l] = comms[i]["k_in"][list(comms[j]["members"])].sum()
				new_comms[k]["k_in"][l] = new_adj[k, l]
				l += 1
			k += 1
		adj = new_adj
		comms = new_comms
		deg = adj.sum(axis=0)
		n_nodes = len(comms)
		node_map = {i: i for i in range(n_nodes)}
		weight_sum = adj.sum() / 2.

# import time
# adj = []
# with open("karate.txt") as f:
# 	l = f.readline()
# 	while l:
# 		adj.append([float(n) for n in l.split(" ")[1:]])
# 		l = f.readline()
# adj = np.array(adj).clip(0, 1)
# start = time.clock()
# print(directed_louvain(adj))
# print(time.clock() - start)

edges[(1,)]
print(directed_louvain(adj))




