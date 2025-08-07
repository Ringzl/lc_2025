class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        
        m, n = len(grid), len(grid[0])
        is_visited = [[False for _ in range(n)] for _ in range(m)]


        dirs = [
            [0,1], [0,-1], [1,0], [-1,0]
        ]



        def dfs(i, j):
            for d in dirs:
                ni = i + d[0]
                nj = j + d[1]
                if ni < 0 or ni >= m or nj < 0 or nj >= n or is_visited[ni][nj]:
                    continue
                
                if grid[ni][nj] == '1':
                    is_visited[ni][nj] = True
                    dfs(ni, nj)


        def bfs(i, j):
            q = deque()
            is_visited[i][j] = True
            q.append((i,j))

            while len(q) > 0:
                x, y = q.popleft()
                for d in dirs:
                    nx = x + d[0]
                    ny = y + d[1]

                    if nx < 0 or nx >= m or ny < 0 or ny >= n or is_visited[nx][ny]:
                        continue

                    if grid[nx][ny] == '1':
                        is_visited[nx][ny] = True
                        q.append((nx, ny))
            
        ans = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1' and not is_visited[i][j]:
                    # dfs(i,j)
                    bfs(i,j)
                    ans += 1

        return ans