class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        ans = []

        cb = ['.'*n for _ in range(n)]

        # 回溯
        def backtrack(n, row, cb):
            if row == n:
                ans.append(cb.copy())
                return

            
            for col in range(n):
                if isValid(row, col, cb, n):
                    cb[row] = cb[row][:col] + 'Q' + cb[row][col+1:]
                    backtrack(n, row+1, cb)
                    cb[row] = cb[row][:col] + '.' + cb[row][col+1:]
        
        # 判断棋盘合法： 在row，col位置放置是否合法
        def isValid(row, col, cb, n):

            # 不同列
            for i in range(row):
                if cb[i][col] == 'Q':
                    return False

            # 对角线（45）
            for i, j in zip(range(row-1, -1, -1), range(col+1, n)):
                if cb[i][j] == 'Q':
                    return False

            # 对角线（135）
            for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
                if cb[i][j] == 'Q':
                    return False
            
            return True
        
        backtrack(n, 0, cb)

        return ans