class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        
        nums.sort()
        n = len(nums)

        ans = []
        for i in range(n):
            # 去重复
            if i > 0 and nums[i] == nums[i-1]:
                continue

            # 双指针
            l = i + 1
            r = n - 1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s == 0:
                    ans.append([nums[i], nums[l], nums[r]]) 
                    while l < r and nums[l] == nums[l+1]:
                        l+=1
                    while l < r and nums[r] == nums[r-1]:
                        r-=1
                    l+=1
                    r-=1
                elif s < 0 :
                    l += 1
                elif s > 0:
                    r -= 1

        return ans 