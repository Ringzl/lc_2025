class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        n = len(heights)
        ans = 0
        for i in range(n):
            h_min = float('inf')
            for j in range(i,n):
                h_min = min(h_min, heights[j])
                ans = max(ans, (j-i+1) * h_min)
        return ans