class Solution:
    def maxArea(self, height: List[int]) -> int:
        n = len(height)

        i, j = 0, n-1
        ans = 0
        while i < j:
            ans = max(ans, (j-i) * min(height[i], height[j]))
            if height[i] > height[j]:
                j -= 1
            else:
                i += 1
        
        return ans 

        