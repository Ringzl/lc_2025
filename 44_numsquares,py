class Solution:
    def numSquares(self, n: int) -> int:
        
        dp = [0 for _ in range(n+1)]
        j = 1
        for i in range(1, n+1):
            dp[i] = i # 最坏情况
            while j * j <= i:
                dp[i] = min(dp[i], dp[i-j*j] + 1)
                j+=1

        return dp[n]