class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        
        n = len(coins)

        dp = [float('inf') for _ in range(amount+1)]
        dp[0] = 0
        for i in range(amount+1):
            for j in range(n):
                if i - coins[j] >= 0:
                    dp[i] = min(dp[i], dp[i-coins[j]] + 1)
        
        return dp[amount] if dp[amount] != float('inf') else - 1