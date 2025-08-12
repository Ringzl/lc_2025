class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:

        m, n = len(grid), len(grid[0])
        cnt = 0
        q = deque()
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    cnt += 1
                elif grid[i][j] == 2:
                    q.appendleft([i,j])
        if cnt == 0:
            return 0

        ans = 0

        dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]]

        while cnt > 0 and len(q) > 0:
            size = len(q)

            for i in range(size):
                x, y = q.pop()

                for d in dirs:
                    nx, ny = x + d[0], y + d[1]

                    # 未超出边界且新鲜
                    if nx >= 0 and nx < m and ny >= 0 and ny < n and  grid[nx][ny] == 1:
                        cnt -= 1
                        grid[nx][ny] = 2
                        q.appendleft([nx,ny])
            ans += 1
        
        if cnt > 0:
            return -1
        else:
            return ans