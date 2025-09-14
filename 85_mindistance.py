class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)

        if m * n == 0:
            return m + n
        
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]

        # 边界
        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j
        

        for i in range(1, m+1):
            for j in range(1, n+1):
                s1 = dp[i-1][j] + 1
                s2 = dp[i][j-1] + 1
                s3 = dp[i-1][j-1]
                if word1[i-1] != word2[j-1]:
                    s3 += 1
                dp[i][j] = min(s1, s2, s3)

        return dp[m][n]