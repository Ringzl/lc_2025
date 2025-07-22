class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        i, j = 0, 0
        n = len(nums)

        while i < n:
            if nums[i] != 0:
                nums[j] = nums[i]
                j += 1 
            i+=1
        
        while j < n:
            nums[j] = 0
            j+=1