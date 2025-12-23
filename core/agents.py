from __future__ import annotations

import random
import heapq
from dataclasses import dataclass
from typing import Callable

from core.grid import Grid
from core.path import Path


@dataclass(slots=True)
class Agent:
    name: str

    def find_path(self, grid: Grid, start: tuple[int, int], goal: tuple[int, int]) -> Path:
        raise NotImplementedError


class ExampleAgent(Agent):

    def __init__(self):
        super().__init__("Example")

    def find_path(self, grid: Grid, start: tuple[int, int], goal: tuple[int, int]) -> Path:
        nodes = [start]
        while nodes[-1] != goal:
            r, c = nodes[-1]
            neighbors = grid.neighbors4(r, c)

            min_dist = min(grid.manhattan(t.pos, goal) for t in neighbors)
            best_tiles = [
                tile for tile in neighbors
                if grid.manhattan(tile.pos, goal) == min_dist
            ]
            best_tile = best_tiles[random.randint(0, len(best_tiles) - 1)]

            nodes.append(best_tile.pos)

        return Path(nodes)


class DFSAgent(Agent):

    def __init__(self):
        super().__init__("DFS")
    @staticmethod
    def directional_cost(a: tuple[int, int], b: tuple[int, int]):
        row_1 = a[0]
        col_1 = a[1]
        row_2 = b[0]
        col_2 = b[1]
        # b istocno od a
        #a.x + 1 == b.x
        if (row_1 == row_2 and col_1 + 1 == col_2): return 0
        # b juzno od a
        if (row_1 == row_2 - 1 and col_1 == col_2): return 1
        # b zapadno od a
        # a.x - 1 == b.x
        if (row_1 == row_2 and col_1 - 1 == col_2): return 2
        # b severno od a
        if (row_1 - 1 == row_2 and col_1 == col_2): return 3
    def dfs_with_path(self, grid: Grid, start: tuple[int, int], goal: tuple[int, int], path: list[tuple[int, int]]):
        if (start == goal):
            return path
        row = start[0]
        col = start[1]
        neighbors = grid.neighbors4(row, col)
        neighbors.sort(key=lambda neighbor: (
        neighbor.cost,
        DFSAgent.directional_cost(start, neighbor.pos)
        ))
        for neighbor in neighbors:
            if neighbor.pos in path:
                continue
            new_path = path.copy()
            new_path.append(neighbor.pos)
            res_path = self.dfs_with_path(grid, neighbor.pos, goal, new_path)
            if (len(res_path) > 0): return res_path
        return []

    def find_path(self, grid: Grid, start: tuple[int, int], goal: tuple[int, int]) -> Path:
        nodes = self.dfs_with_path(grid, start, goal, [start])
        return Path(nodes)



class BranchAndBoundAgent(Agent):

    def __init__(self):
        super().__init__("BranchAndBound")

    def find_path(self, grid: Grid, start: tuple[int, int], goal: tuple[int, int]) -> Path:
        leaf_nodes = []
        heapq.heapify(leaf_nodes)
        popped = start
        best_path = None
        INF = 2 ** 31 - 1
        best_goal_cost = INF
        best_costs = [[INF for _ in range(grid.cols)] for _ in range(grid.rows)]

        for neighbor in grid.neighbors4(start[0], start[1]):
            heapq.heappush(leaf_nodes, (neighbor.cost, 2, neighbor.pos,  [start, neighbor.pos]))

        while(leaf_nodes):
            data = heapq.heappop(leaf_nodes)
            pos = data[2]
            cost = data[0]
            path = data[3]
            if cost < best_costs[pos[0]][pos[1]]:
                best_costs[pos[0]][pos[1]] = cost
            else:
                continue
            if cost >= best_goal_cost:
                continue
            if pos == goal:
                best_goal_cost = cost
                best_path = path
                return Path(best_path)
            else:
                for neighbor in grid.neighbors4(pos[0], pos[1]):
                    if neighbor.pos in path:
                        continue
                    if cost + neighbor.cost >= best_costs[neighbor.row][neighbor.col]:
                        continue
                    new_path = path.copy()
                    new_path.append(neighbor.pos)
                    heapq.heappush(leaf_nodes, (cost + neighbor.cost, len(new_path), neighbor.pos, new_path))
        return Path(best_path)

class AStar(Agent):

    def __init__(self):
        super().__init__("AStar")
    @staticmethod
    def heuristic(start: tuple[int, int], goal: tuple[int, int]):
        return abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    def find_path(self, grid: Grid, start: tuple[int, int], goal: tuple[int, int]) -> Path:
        leaf_nodes = []
        heapq.heapify(leaf_nodes)
        popped = start
        best_path = None
        INF = 2 ** 31 - 1
        best_goal_cost = INF
        best_costs = [[INF for _ in range(grid.cols)] for _ in range(grid.rows)]

        for neighbor in grid.neighbors4(start[0], start[1]):
            heapq.heappush(leaf_nodes, (neighbor.cost + AStar.heuristic(neighbor.pos, goal), neighbor.cost,  2, neighbor.pos, [start, neighbor.pos]))

        while (leaf_nodes):
            data = heapq.heappop(leaf_nodes)
            pos = data[3]
            cost = data[1]
            path = data[4]
            if cost < best_costs[pos[0]][pos[1]]:
                best_costs[pos[0]][pos[1]] = cost
            else:
                continue
            if cost >= best_goal_cost:
                continue
            if pos == goal:
                best_goal_cost = cost
                best_path = path
                return Path(best_path)
            else:
                for neighbor in grid.neighbors4(pos[0], pos[1]):
                    if neighbor.pos in path:
                        continue
                    if cost + neighbor.cost >= best_costs[neighbor.row][neighbor.col]:
                        continue
                    new_path = path.copy()
                    new_path.append(neighbor.pos)
                    heapq.heappush(leaf_nodes, (cost + neighbor.cost + AStar.heuristic(neighbor.pos, goal), cost + neighbor.cost , len(new_path), neighbor.pos, new_path))
        return Path(best_path)


AGENTS: dict[str, Callable[[], Agent]] = {
    "Example": ExampleAgent,
    "DFS": DFSAgent,
    "BranchAndBound": BranchAndBoundAgent,
    "AStar": AStar
}


def create_agent(name: str) -> Agent:
    if name not in AGENTS:
        raise ValueError(f"Unknown agent '{name}'. Available: {', '.join(AGENTS.keys())}")
    return AGENTS[name]()
