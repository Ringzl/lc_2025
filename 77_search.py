class Solution:
    def search(self, nums: List[int], target: int) -> int:
        
        n = len(nums)

        if n == 0:
            return -1

        l = 0
        r = n-1
        while l <= r:

            mid = (l+r) //2

            if nums[mid] == target:
                return mid
            
            # 左边有序
            if nums[0] <= nums[mid]:
                if nums[0] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                if nums[mid] < target <= nums[n-1]:
                    l = mid + 1
                else:
                    r = mid - 1
            
        return -1 