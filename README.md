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

date： 20250723
### 5. 无重复字符的最长子串

Problem: 给定一个字符串 s ，请你找出其中不含有重复字符的 最长 子串 的长度。

Think:
1. 遍历元素，同时用字典保存之前遍历过的元素及其位置，若之前遍历过，则将遍历过的位置之前的元素清掉
2. ans = max(ans, len(char_dct))

Solution:
```py
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        
        # 保存位置
        char_dct = {}
        n = len(s)
        ans = 0
        for i in range(n):

            # 存在重复，找到重复位置，并从字典中去除该位置前的字符
            if s[i] in char_dct:
                pos = char_dct[s[i]]
                for c in list(char_dct.keys()):
                    if c in char_dct and char_dct[c] <= pos:
                        char_dct.pop(c)
            
            char_dct[s[i]] = i

            ans = max(ans, len(char_dct))
        return ans
```

Think2:
1. 使用元组保存当前字串元素，使用left、right记录子串左右侧位置，通过right遍历
2. s[right] not in char_set -> 添加、right +=1 cnt+=1
3. 否则 char_set.remove(s[left]) -> 删除（删到相同值位置为止）、left += 1 cnt -= 1


Solution2:
```py

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        left = 0
        right = 0
        n = len(s)

        char_set = set()
        ans = 0
        cnt = 0
        while right < n:
            if s[right] not in char_set:
                char_set.add(s[right])
                right += 1
                cnt += 1
            else:
                char_set.remove(s[left])
                left += 1
                cnt -= 1
            ans = max(ans, cnt)

        return ans
```


### 6. 合并区间

Problem: 以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。

Think:
1. 先按 start 排序

Solution:
```py
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:

        intervals.sort(key=lambda x:x[0])

        n = len(intervals)
        ans = [intervals[0]]

        for i in range(1, n):
            if ans[-1][1] < intervals[i][0]:
                ans.append(intervals[i])
            else:
                # 合区间
                ans[-1][1] = max(intervals[i][1], ans[-1][1])
        return ans
```

date: 20250724
### 7. 盛最多水的容器

Problem:给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。返回容器可以储存的最大水量。

Think:
1. 从首尾两侧左边i,右边j开始，确定移动左侧还是右侧？ 宽度在减小，高度尽可能高 -> 保持高的边不动
2. 容量计算 (j-i) * min(height[i], height[j])

Solution:
```py
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
```

### 8. 搜索插入位置

Problem:给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

Think:
1. 二分法查找元素位置，找到位置和插入位置-> 找 nums[mid] >= target 等于或大于的位置

Solution:
```py
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums)-1
        
        pos = len(nums)
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] >= target:
                pos = mid
                right = mid - 1
            else:
                left = mid + 1

        return pos
```

DATE:20250725
### 9. 每日温度
Problem: 给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，其中 answer[i] 是指对于第 i 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 0 来代替。

Think:
1. 暴力遍历所有情况（超时）
2. 维护一个存储下标的单调栈，栈顶到栈底温度递减；遍历当前温度大于栈顶，则栈顶出栈


Solution1:
```py
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        
        n = len(temperatures)

        ans = [0 for _ in range(n)]
        for i in range(n-1):
            for j in range(i+1, n):
                if temperatures[j] > temperatures[i]:
                    ans[i] = j-i
                    break
        return ans
```

Solution2:
```py
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        ans = [0 for _ in range(n)]
        st = []
        for i in range(n):
            while st and temperatures[i] > temperatures[st[-1]]:
                ans[st[-1]] = i - st[-1]
                st.pop()
            st.append(i)
        return ans 
```

### 10. 反转链表

Problem: 给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。

Think: 
1. 保存前后两个节点prev、cur
2. tmp = cur.next; cur.next = prev; prev = cur; cur = tmp;

Solution1:
```py

class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        
        prev = None
        cur = head
        while cur != None:
            tmp = cur.next
            cur.next = prev
            prev = cur
            cur = tmp
        
        return prev
```

### 11. 多数元素

Problem: 给定一个大小为 n 的数组 nums ，返回其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。

Think: 
1. 使用字典记录元素出现次数，并更新最大次数和对应元素值

Solution:

```py
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        cnt_dct = {}
        n = len(nums)
        max_cnt = 0 
        ans = 0
        for i in range(n):
            cnt_dct[nums[i]] = cnt_dct.get(nums[i], 0) + 1
            if cnt_dct[nums[i]] > max_cnt:
                max_cnt = cnt_dct[nums[i]]
                ans = nums[i]
        return ans 
```