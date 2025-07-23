# lc_2025

date： 20250721
### 1. 两数之和
Problem: 给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。

Think: 
1. 找到两个数 a 和 b 在数组的位置, 且 a + b == target
2. 即找 a 和 target - a，遍历一次找到符合条件的 a及位置
3. 用一个字典保存数组值和位置

Solution: 
```py
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        num_pos_dict = {}
        for i,num in enumerate(nums):
            if target - num in num_pos_dict and num_pos_dict[target - num] != i:
                return i, num_pos_dict[target - num]
            num_pos_dict[num] = i 
```

date： 20250722
### 2. 移动零
Problem: 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
请注意 ，必须在不复制数组的情况下原地对数组进行操作。

Think:
1. 使用 i 遍历数组位置，j 记录非0位置
2. 最后数组 j 位置后补0

Solution:
```py
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
```

### 3. 字母异位分组
Problem: 给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。

Think:
1. 将每一个字符串排序后，作为key,value为对应字符串数组
2. 遍历字典取value到数组

Solution:
```py
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        dct = {}
        for s in strs:
            sorted_s = ''.join(sorted(s))
            dct.setdefault(sorted_s, []).append(s)
        
        ans = []
        for key in dct:
            ans.append(dct[key])

        return ans        
```


### 4. 最长递增子序列
Problem: 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。

Think1:

1. 字数组的最长递增子序列长度可以可以用dp解决，用 i,j 表示递增相邻位置
2. if nums[j] > nums[i] -> dp[j] = max(dp[j], dp[i] + 1)  

Solution1:
```py
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        
        n = len(nums)
        dp = [1 for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if nums[j] > nums[i]:
                    dp[j] = max(dp[j], dp[i] + 1)  
        return max(dp) 
```

Think2:
1. 要使递增子序列尽可能长，需要让序列上升得尽可能慢（每次加上的数尽可能小）
2. 将原数组当作key,状态数组作为value,问题变成了在集合中寻找key小于目标值的最大value
3. 维护一个数组d保存最长递增子序列末尾元素最小值（可证明d单调递增）
4. 在d中利用二分法找到第一个比 nums[i] 小的数 d[k] ，并更新 d[k+1]=nums[i]


Solution2:
```py
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        
        n = len(nums)
        if n == 0:
            return 0

        def binarysearch_le(d, l, r, target):
            pos = 0
            while l <= r:
                mid = (l + r) // 2
                if d[mid] < target:
                    pos = mid
                    l = mid + 1
                else:
                    r = mid - 1
            return pos

        k = 1
        d = [0 for _ in range(n+1)]
        d[k] = nums[0] 

        for i in range(1,n):
            if nums[i] > d[k]:
                d[k+1] = nums[i]
                k+=1
            else:
                pos = binarysearch_le(d, 1, k, nums[i])
                d[pos+1] = nums[i]
        
        return k
```


