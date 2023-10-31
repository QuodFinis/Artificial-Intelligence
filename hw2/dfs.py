# creating dfs algorithm for a tree
# dfs uses a stack for considering which node to traverse first

# the graph will be represented using an adjacency list

# the dfs algorithm will be given a adjacency list, and a start and end state.
# dfs works by adding all adjacent nodes to the stack and exploring the stack using FIFO
# we need to make sure we do not repeat nodes so keep track using list of visited nodes
# the purpose of this is to see the path, but notice this is just the visited nodes

## i added visited so that this can be an iterative process, or i can just create an iter
## i want to call dfs with the new node as the start and same end but updated visited

def dfs(start, end, adj_list, visited):
    visited = [start]

    # frontier - simple stack by using list, append, and pop
    stack = []

    # adding each adj node to the stack
    for node in adj_list[start]:
        stack.append(node)

    # removing the last added node and visiting it
    visited.append(stack.pop)

    # now i continue dfs(

    return visited


# another way to keep track can be to make a class
# actually the best way is to use   while stack is not empty
# keep running if there is more to explore

def dfs(start: str, end: str, adjacency_list: dict) -> list:
    visited = []
    stack = [start]

    for node in adjacency_list[start]:
        stack.append(node)

    while stack:
        current_node = stack.pop()
        visited.append(current_node)

        if current_node == end:
            return visited

    return visited


# let's see what the path looks like
# print(dfs(start='1', end='15', adjacency_list=adj_list))

# i forgot to make sure nodes are not repeated so i also need to include this check.
# the stack is also inefficent. we can just initialize the stack with the start node
# and then pop it and check if it is visited. then we can add it to visited.
# also the adj_list was incorrectly created, set should not be use it is also corrected

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


def dfs(start: str, end: str, adjacency_list: dict) -> list:
    visited = []
    stack = [start]

    while stack:
        current_node = stack.pop()
        if current_node not in visited:
            visited.append(current_node)

        if current_node == end:
            return visited

        for neighbor in adjacency_list[current_node]:
            if neighbor not in visited and neighbor not in stack:
                stack.append(neighbor)

        print(visited, stack)

    return visited


# let's see what the path looks like
print(dfs(start='1', end='15', adjacency_list=adj_list))
# ['1', '6', '11', '12', '13', '14', '15']

# this path is simllar to what I had expected 1-2-3-4-5-10-15
# the reason it takes the bottom path is that the largest number is
# always added to the end so it takes the route along the bottom
# if i wanted the route along the top i could reverse the order of the
# numbers in the adjacency list to be greatest to least so least is explored

'''
lets trace it
visited | stack
['1']                               |  ['2', '6']
['1', '6']                          |  ['2', '7', '11']
['1', '6', '11']                    |  ['2', '7', '12']
['1', '6', '11', '12']              |  ['2', '7', '13']
['1', '6', '11', '12', '13']        |  ['2', '7', '8', '14']
['1', '6', '11', '12', '13', '14']  |  ['2', '7', '8', '9', '15']
'''

