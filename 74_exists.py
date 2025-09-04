class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:


        def dfs(board, i, j, word, index, is_visited):

            if index == len(word):
                return True
            
            if i<0 or i>=len(board) or j<0 or j>=len(board[0]) or board[i][j] != word[index] or is_visited[i][j]:
                return False  
            
            is_visited[i][j] = True
            found = (
                dfs(board, i+1, j, word, index+1, is_visited) or 
                dfs(board, i-1, j, word, index+1, is_visited) or 
                dfs(board, i, j+1, word, index+1, is_visited) or 
                dfs(board, i, j-1, word, index+1, is_visited)
            )    
            is_visited[i][j] = False

            return found

        m, n = len(board), len(board[0])
        is_visited = [[False for _ in range(n)] for _ in range(m)]

        for i in range(m):
            for j in range(n):
                if dfs(board, i, j, word, 0, is_visited):
                    return True
        
        return False