class Solution:
    def longestPalindrome(self, s: str) -> str:
        
        n = len(s)
        dp = [[0 for _ in range(n)] for _ in range(n)]

        if n == 0 or n == 1:
            return s

        max_l = 1
        start = 0 
        # 初始化
        for i in range(n):
            dp[i][i] = 1
            if i < n-1 and s[i] == s[i+1]:
                dp[i][i+1] = 1
                start = i
                max_l = 2
                

        for l in range(3, n+1):

            for i in range(n-l+1):
                j = i + l -1

                if s[i] == s[j] and dp[i+1][j-1] == 1:
                    dp[i][j] = 1
                    start = i
                    max_l = l
        
        return s[start:start+max_l]