import csv
import heapq
edgeFile = 'edges.csv'
"""
Save the file 'edges.csv' into the dictionary "graph".
Every row may like [start, end, dist, limit].
Store it into dictionary as: 'start':[end, distance]
"""
graph = {}
with open(edgeFile) as file:
    csvfile = list(csv.reader(file))
    csvfile.pop(0)
    for i in csvfile:
        num1, num2, distance = int(i[0]), int(i[1]), float(i[2])
        if num1 not in graph:
            graph[num1] = [(num2, distance)]
        else:
            graph[num1].append((num2, distance))
        if num2 not in graph:
            graph[num2] = []

def ucs(start, end):
    # Begin your code (Part 3)
    """
    Build the dictionary "parent" to store the parent of store, in order to trace back the path at last.
    Build a heapified list like priority queue called "heap" storing tuple (dist, node, parent) in the heap.
    Pop the first tuple of the heap.
    If the node haven't been visited, cnt++, record the node's parent, and get the node's neighbor and push into heap.
    If the node == end, then we find the path and the cost(distance).
    According to "parent", we record the path into "path" and return the path, dist, cnt.
    """
    parent = {}
    heap = [(0, start, None)]
    heapq.heapify(heap)
    visited = set()
    cnt, isfind, dist = 0, 0, 0
    while heap:
        (cost, node, p) = heapq.heappop(heap)
        if node == end:
            dist = cost
            parent[node] = p
            path=[]
            path.append(end)
            while path[-1]!=start:
                path.append(parent[path[-1]])
            path.reverse()
            return path, dist, cnt
        if node not in visited:
            cnt+=1
            visited.add(node)
            parent[node] = p
            for en, dis in graph[node]:
                heapq.heappush(heap, (cost + dis, en, node))
    # raise NotImplementedError("To be implemented")
    # End your code (Part 3)


if __name__ == '__main__':
    path, dist, num_visited = ucs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
