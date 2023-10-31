adj_list = {
    '1': ['2', '6'],
    '2': ['1', '3', '7'],
    '3': ['2', '4', '8'],
    '4': ['3', '5', '9'],
    '5': ['4', '10'],
    '6': ['1', '7', '11'],
    '7': ['2', '6', '8', '12'],
    '8': ['3', '7', '9', '13'],
    '9': ['4', '8', '10', '14'],
    '10': ['5', '9', '15'],
    '11': ['6', '12'],
    '12': ['7', '11', '13'],
    '13': ['8', '12', '14'],
    '14': ['9', '13', '15'],
    '15': ['10', '14'],
}

# bfs should explore using a queue but will still be structually similar to dfs
# a queue can be implemented using list by popping the 0th element instead of the last

def bfs(start: str, end: str, adjacency_list: dict) -> list:
    visited = []
    stack = [start]

    while stack:
        current_node = stack.pop(0)
        if current_node not in visited:
            visited.append(current_node)

        if current_node == end:
            return visited

        for neighbor in adjacency_list[current_node]:
            if neighbor not in visited and neighbor not in stack:
                stack.append(neighbor)

        # print(visited, stack)

    return visited


# let's see what the path looks like
print(bfs(start='1', end='15', adjacency_list=adj_list))
# ['1', '2', '6', '3', '7', '11', '4', '8', '12', '5', '9', '13', '10', '14', '15']

# this path is what i expected. it looks like a search radially out from 1 explored every node