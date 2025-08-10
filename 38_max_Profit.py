class Solution:
    def maxProfit(self, prices: List[int]) -> int:

        ans = 0
        n = len(prices)

        min_price = float('inf')
        for i in range(n):
            if prices[i] < min_price:
                min_price = prices[i]
            else:
                ans = max(ans, prices[i] - min_price )

        return ans