import csv
edgeFile = 'edges.csv'
"""
Save the file 'edges.csv' into the dictionary "graph".
Every row may like [start, end, dist, limit].
Store it into dictionary as: 'start':[end, distance]
"""
graph={}
with open(edgeFile, newline='') as file:
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

def bfs(start, end):
    # Begin your code (Part 1)
    """
    Build a queue to store the path with [current_vertex, [path]].
    Every time pop the fisrt element in queue and find its neighbor nodes.
    If we haven't visited the node, cnt++ and append it into queue.
    If the node == end, then we get the path from start to end and we can calculate the distance.
    Add the distance corresponding to the dictionary 'graph', and add each distance one by one.
    After all, we return the path, dist, and cnt.
    """
    dist=0
    cnt=1
    queue=[(start, [start])]
    visited=set()
    while queue:
        vertex, path = queue.pop(0)
        visited.add(vertex)
        for node in graph[vertex]:
            if node[0] == end:
                path += [end]
                st=start
                for v in path:      # st->v
                    if v==st:   continue
                    for n in graph[st]:
                        if n[0]==v:
                            dist += n[1]
                            break
                    st=v
                return path, dist, cnt
            else:
                if node[0] not in visited:
                    cnt+=1
                    visited.add(node[0])
                    queue.append((node[0], path+[node[0]]))    
    
    # raise NotImplementedError("To be implemented")
    # End your code (Part 1)


if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
