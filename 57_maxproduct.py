class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        
        n = len(nums)
        pos = max(0, nums[0])
        neg = min(0, nums[0])

        ans = nums[0]
        for i in range(1, n):
            
            if nums[i] < 0:
                pos_tmp = pos
                pos = max(nums[i], nums[i] * neg)
                neg = min(nums[i], nums[i] * pos_tmp)
            else:
                pos = max(nums[i], nums[i] * pos)
                neg = min(nums[i], nums[i] * neg)
            
            ans = max(ans, pos)

        return ans