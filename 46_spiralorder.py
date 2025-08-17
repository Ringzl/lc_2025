class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:

        m, n = len(matrix), len(matrix[0])

        # # 顺序 右、下、左、上
        # dirs = [
        #     [0,1], [1,0], [0,-1], [-1,0]
        # ]

        # is_visited = [
        #     [False for _ in range(n)] for _ in range(m)
        # ]

        # total = m * n

        # row, col = 0,0
        # dir_index = 0 # 初始方向
        # ans = []
        # for i in range(total):
        #     ans.append(matrix[row][col])
        #     is_visited[row][col] = True

        #     next_row, next_col = row + dirs[dir_index][0], col + dirs[dir_index][1]

        #     if next_row < 0 or next_row >= m or next_col < 0 or next_col >= n or is_visited[next_row][next_col]:
        #         dir_index = (dir_index + 1) % 4

        #     row += dirs[dir_index][0]
        #     col += dirs[dir_index][1]

        # 分层模拟
        left, right, top, bottom = 0, n-1, 0, m-1

        ans = []
        while left <= right and top <= bottom:
            
            # 从左到右
            for i in range(left, right+1):
                ans.append(matrix[top][i])
            top += 1

            # 从上到下
            for i in range(top, bottom+1):
                ans.append(matrix[i][right])
            right -= 1

            # 从右到左
            if top <= bottom:
                for i in range(right, left-1, -1):
                    ans.append(matrix[bottom][i])
            bottom -= 1
            
            # 从下到上
            if left <= right:
                for i in range(bottom, top-1, -1):
                    ans.append(matrix[i][left])
            left += 1

        return ans



            




        