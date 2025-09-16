class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        # 从后向前查找第一个顺序对 
        i = len(nums) - 2
        while i >=0  and nums[i] >= nums[i+1]:
            i -= 1
        
        # 从后向前查找第一个元素 j 满足 a[i] < a[j], 交换 a[i] 与 a[j]
        if i >= 0:
            j = len(nums) - 1
            while j >= 0 and nums[i] >= nums[j]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]

        # 区间 [i+1,n) 必为降序, 使用双指针反转区间 [i+1,n) 使其变为升序
        left, right = i+1, len(nums) - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1 
