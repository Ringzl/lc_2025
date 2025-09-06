class Solution:
    def partition(self, s: str) -> List[List[str]]:
        n = len(s)
        is_huiwen = [
            [True for _ in range(n)] for _ in range(n)
        ]
        for i in range(n-1, -1, -1):
            for j in range(i+1, n):
                is_huiwen[i][j] = (is_huiwen[i+1][j-1] and s[i] == s[j])

        ret = []
        ans = []
        def dfs(s, i):
            if i == n:
                ret.append(ans.copy())
                return 

            for j in range(i, n):
                if is_huiwen[i][j]:
                    ans.append(s[i:j+1])
                    dfs(s, j+1)
                    ans.pop()
        
        dfs(s, 0)

        return ret