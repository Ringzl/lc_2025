class Solution:
    def canPartition(self, nums: List[int]) -> bool:

        n = len(nums)

        if n < 2:
            return False
        
        sum_num = sum(nums)
        max_num = max(nums)
        target = sum_num // 2
        if sum_num % 2 != 0 or max_num > target:
            return False
        
        dp = [[False] * (target + 1) for _ in range(n)]
        for i in range(n):
            dp[i][0] = True
        dp[0][nums[0]] = True
        for i in range(1, n):
            for j in range(1, target + 1):
                if j >= nums[i]:
                    dp[i][j] = dp[i-1][j] or dp[i-1][j-nums[i]]
                else:
                    dp[i][j] = dp[i-1][j]

        return dp[n-1][target]