from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

from backend.sql.schema import SchemaInfo


# Graph edge type: (neighbor_table, local_col, neighbor_col)
Edge = Tuple[str, str, str]


def build_schema_graph(schema_info: SchemaInfo) -> Dict[str, List[Edge]]:
    """Build an undirected adjacency list from SchemaInfo foreign keys."""
    graph = defaultdict(list)
    
    for tname, tinfo in schema_info.tables.items():
        # foreign_keys: List[Tuple[str, str, str]]
        # (local_col, referenced_table, referenced_col)
        for local_col, ref_table, ref_col in tinfo.foreign_keys:
            # Directed edge: tname -> ref_table
            graph[tname].append((ref_table, local_col, ref_col))
            # Undirected edge: ref_table -> tname
            graph[ref_table].append((tname, ref_col, local_col))
            
    return graph


def find_shortest_path(
    graph: Dict[str, List[Edge]], 
    start_nodes: Set[str], 
    target_node: str
) -> Tuple[List[str], List[str]]:
    """
    Find shortest path from any node in start_nodes to target_node using BFS.
    Returns:
      (path_tables, join_conditions)
      where path_tables is a list of tables to traverse,
      and join_conditions is a list of "JOIN t2 ON t1.c1 = t2.c2" strings.
    """
    if target_node in start_nodes:
        return ([target_node], [])
        
    # Queue stores: (current_node, path_tables, join_conditions)
    queue = deque()
    visited = set(start_nodes)
    
    for start_node in start_nodes:
        queue.append((start_node, [start_node], []))
        
    while queue:
        curr_node, path_tables, join_conds = queue.popleft()
        
        if curr_node == target_node:
            return path_tables, join_conds
            
        for neighbor, local_col, neighbor_col in graph[curr_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path_tables)
                new_path.append(neighbor)
                
                new_conds = list(join_conds)
                # Format: "JOIN neighbor ON curr_node.local_col = neighbor.neighbor_col"
                new_conds.append(f"JOIN {neighbor} ON {curr_node}.{local_col} = {neighbor}.{neighbor_col}")
                
                queue.append((neighbor, new_path, new_conds))
                
    # If no path found
    return ([], [])


def compute_join_tree(schema_info: SchemaInfo, selected_tables: List[str]) -> Tuple[Set[str], str]:
    """
    Given a set of tables, find a spanning tree of foreign key joins connecting them.
    Returns:
        (all_required_tables_set, join_string)
    """
    if not selected_tables:
        return set(), ""
        
    if len(selected_tables) == 1:
        return {selected_tables[0]}, f"FROM {selected_tables[0]}"
        
    graph = build_schema_graph(schema_info)
    
    # Build the tree iteratively using a greedy approach.
    # Start with the first table.
    connected_tables = {selected_tables[0]}
    join_string = f"FROM {selected_tables[0]}"
    all_required_tables = {selected_tables[0]}
    
    for target in selected_tables[1:]:
        if target in connected_tables:
            continue
            
        path_tables, join_conds = find_shortest_path(graph, connected_tables, target)
        if not path_tables:
            # Cannot connect to target table (disconnected graph component).
            # Fallback to CROSS JOIN (or comma separation).
            join_string += f" CROSS JOIN {target}"
            connected_tables.add(target)
            all_required_tables.add(target)
            continue
            
        # Add the newly discovered path to our join string
        for cond in join_conds:
            join_string += f" {cond}"
            
        # Update connected components with all intermediate bridge tables and the target
        for t in path_tables:
            connected_tables.add(t)
            all_required_tables.add(t)
            
    return all_required_tables, join_string
