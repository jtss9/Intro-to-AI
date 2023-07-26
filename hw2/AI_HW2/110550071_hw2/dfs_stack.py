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


def dfs(start, end):
    # Begin your code (Part 2)
    """
    Build a stack to store the path with [current_vertex, [path]].
    Every time we pop the last element in queue and get the vertex.
    Check if we have visited the vertex before, skip if we have visited.
    If the vertex is not the end, cnt++ and get the neighbor of the node and append it to the stack.
    If the vertex == end, then we get the oath from start to end and we can calculate the distance.
    Add the distance corresponding to the dictionary 'graph', and add each distance one by one.
    After all, we return the path, dist, and cnt.
    """
    dist=0
    cnt=0
    stack = [(start, [start])]
    visited = set()   
    while stack:
        vertex, path = stack.pop()
        if vertex not in visited:
            if vertex == end:
                st=start
                for v in list(map(int, path)):      # st->v
                    if v==st:   continue
                    for n in graph[st]:
                        if n[0]==v:
                            dist += n[1]
                            break
                    st=v
                return path, dist, cnt
            visited.add(vertex)
            cnt+=1
            for node in graph[vertex]:
                stack.append((node[0], path+[node[0]]))
    # raise NotImplementedError("To be implemented")
    # End your code (Part 2)


if __name__ == '__main__':
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
