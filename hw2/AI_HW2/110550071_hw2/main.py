import folium
import pickle
def load_path_graph(path):
    with open('graph.pkl', 'rb') as f:
        graph = pickle.load(f)
    node_pairs = list(zip(path[:-1], path[1:]))
    lines = []
    for edge in graph:
        if (edge['u'], edge['v']) in node_pairs or  (edge['v'], edge['u']) in node_pairs:
            lines.append(edge['geometry'])
        #print(edge['u'], edge['v'])
    return lines

# Part 5
# Change start ID and end ID.
start = 1718165260
end = 8513026827

# Part 1
from bfs import bfs
# Don't change this part.
# Show the result of BFS
bfs_path, bfs_dist, bfs_visited = bfs(start, end)
print(f'The number of nodes in the path found by BFS: {len(bfs_path)}')
print(f'Total distance of path found by BFS: {bfs_dist} m')
print(f'The number of visited nodes in BFS: {bfs_visited}\n')

fmap = folium.Map(location=(24.806383132251874, 120.97685775516189), zoom_start=13)
for line in load_path_graph(list(map(int, bfs_path))):
    fmap.add_child(folium.PolyLine(locations=line, tooltip='bfs', weight=4, color='blue'))
fmap.save('bfs.html')

# Part 2
from dfs_stack import dfs
# Don't change this part.
# Show the result of DFS
dfs_path, dfs_dist, dfs_visited = dfs(start, end)
print(f'The number of nodes in the path found by DFS: {len(dfs_path)}')
print(f'Total distance of path found by DFS: {dfs_dist} m')
print(f'The number of visited nodes in DFS: {dfs_visited}\n')

fmap = folium.Map(location=(24.806383132251874, 120.97685775516189), zoom_start=13)
for line in load_path_graph(list(map(int, dfs_path))):
    fmap.add_child(folium.PolyLine(locations=line, tooltip='dfs', weight=4, color='green'))
fmap.save('dfs.html')

# Part 3
from ucs import ucs
# Don't change this part.
# Show the result of UCS
ucs_path, ucs_dist, ucs_visited = ucs(start, end)
print(f'The number of nodes in the path found by UCS: {len(ucs_path)}')
print(f'Total distance of path found by UCS: {ucs_dist} m')
print(f'The number of visited nodes in UCS: {ucs_visited}\n')

fmap = folium.Map(location=(24.806383132251874, 120.97685775516189), zoom_start=13)
for line in load_path_graph(list(map(int, ucs_path))):
    fmap.add_child(folium.PolyLine(locations=line, tooltip='ucs', weight=4, color='violet'))
fmap.save('ucs.html')

# Part 4
from astar import astar
# Don't change this part.
# Show the result of A* search
astar_path, astar_dist, astar_visited = astar(start, end)
print(f'The number of nodes in the path found by A* search: {len(astar_path)}')
print(f'Total distance of path found by A* search: {astar_dist} m')
print(f'The number of visited nodes in A* search: {astar_visited}\n')

fmap = folium.Map(location=(24.806383132251874, 120.97685775516189), zoom_start=13)
for line in load_path_graph(list(map(int, astar_path))):
    fmap.add_child(folium.PolyLine(locations=line, tooltip='astar', weight=4, color='red'))
fmap.save('astar.html')

