import csv
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'
from queue import PriorityQueue
"""
Read the file 'edges.csv' and convert to list and store it into "edges_list", then make a copy called "all_rows".
Pop out the first title row and change the other data into following format:
r: [start, end, dist, limit, which round, parent_start, parent_end, h()]
"""
with open(edgeFile, newline='') as file: # open edge data
        edge_list = list(csv.reader(file))
        all_rows = edge_list
        all_rows.pop(0) #get rid of the first line 
        for r in all_rows:
            # each row : [start,end,distance,speed limit,found @ which round, parent start, parent end, h]
            r[0] = int(r[0]) 
            r[1] = int(r[1])
            r[2] = float(r[2])
            r.append(int(0)) 
            r.append(int(0))
            r.append(int(0))
            r.append(int(0))

def astar(start, end):
    # Begin your code (Part 4)
    """
    Because of different end node, we use 3 key to correspond to 'heuristic.csv'
    Build a list to store h(n) of 3 different end, and append the value of [node, h()]
    Update the h data in 'edge_list'.
    Build a priority queue "bfs_q" and put the [dist+h(), row] of start.
    """
    cnt=0
    node_dist=[]        # h(n)
    if end==1079387396: key=1
    elif end==1737223506:   key=2
    elif end==8513026827:   key=3
    
    with open(heuristicFile, newline='') as f:
        d=csv.reader(f)
        n=list(d)
        n.pop(0)
        for row in n:
            node_dist.append([int(row[0]),float(row[key])])
    for r in edge_list:
        for n in node_dist:     #n: node, h()
            if n[0]==r[1]:
                r[7] = n[1]
                break
    
    bfs_q = PriorityQueue()     # g+h, (start, end, dist, limit)
    for r in edge_list:
        if r[0]==start and r[4]==0:
            cnt+=1
            r[4]=1
            bfs_q.put([r[2]+r[7], r])
    """
    While bfs_q is not empty and we haven't found the end:
        Get the highest priority element from "bfs_q" and get the current location.
        Found the location_node in "edge_list" and record the found_round, parent_start, parent_end.
        And then put the new node [original_priority-h(parent)+new_dist+new_h, r] into priority_queue.
        If we find the end then we record the cnt and break. 
    """   
    dest=[]
    find=False
    while (not bfs_q.empty()) and find==False:
        c = bfs_q.get()
        cur = c[1]
        location = cur[1]
        
        for r in edge_list:
            if r[0]==location and r[4]==0:
                cnt+=1
                r[4]=cur[4]+1
                r[5]=cur[0]
                r[6]=cur[1]
                bfs_q.put([(c[0]-cur[7]+r[2]+r[7]),r])
                if r[1]==end:
                    dest = r
                    find = True
                    break
    """
    We have recorded the dest so that we can trace back the path by parent_start and parent_end.
    After we trace back to start, we append only the node number of the node into "path" and calculate the distance.
    After all, return the path, dist, cnt.
    """
    d = [dest]
    curr = dest
    while curr[0]!=start:
        for r in edge_list:
            if r[0]==curr[5] and r[1]==curr[6]:
                d.append(r)
                curr=r
                break
    d.reverse()
    path=[start]
    dist=0
    for r in d:
        path.append(r[1])
        dist+=r[2]
    return path, dist, cnt
    # raise NotImplementedError("To be implemented")
    # End your code (Part 4)


if __name__ == '__main__':
    path, dist, num_visited = astar(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
