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
2. Boyer-Moore 投票算法，选择一个数为候选众数，遍历后续的数，如果与它相等，count加一，否则减一;count等于0,更换候选。

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

### 12. 最长连续序列

Problem: 给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

Think: 
1. 先排序，然后遍历对比前后元素，差值为1计数加1，相同不变，其他则置1
2. 先用元组去重，遍历num,若num-1不在元组中，则向后找num+1...,同时计数更新最长

Solution:

```py
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:

        nums.sort()
        n = len(nums)
        if n <= 1:
            return n
        ans = 1
        cnt = 1
        for i in range(1, n):
            if nums[i] - nums[i-1] == 1:
                cnt += 1
            elif nums[i] == nums[i-1]:
                continue
            else:
                cnt = 1
            ans = max(ans, cnt)
            
        return ans
```

Sloution2：
```py
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:

        nums_set = set(nums)
        ans = 0
        for num in nums_set:
            if num-1 not in nums_set:
                cur_num = num
                cnt = 1

                while cur_num + 1 in nums_set:
                    cur_num += 1
                    cnt += 1
                ans = max(ans, cnt)
        
        return ans    
```

### 13.回文链表

Problem: 给你一个单链表的头节点 head ，请你判断该链表是否为回文链表。如果是，返回 true ；否则，返回 false 。

Think: 
1. 遍历链表，先进栈，再遍历出栈，对比相同返回True


Sloution：
```py   
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:

        st = []
        cur = head
        while cur != None:
            st.append(cur.val)
            cur = cur.next
        
        cur = head
        while cur != None:
            if cur.val != st[-1]:
                return False
            st.pop()
            cur = cur.next

        return True
```

### 14. 和为 K 的子数组

Problem: 给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的子数组的个数 

Think: 
1. 暴力： i,j 字串和为k（超时）
2. 利用前缀和， presum - k 在前缀和中多少次，既有 presum-(presum-k) = k, 注意初始 presum_dct = {0:1}

Sloution：
```py   
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:

        n = len(nums)
        ans = 0
        for i in range(n):
            for j in range(i, n):
                if sum(nums[i:j+1]) == k:
                    ans += 1

        return ans
```

Sloution2：
```py   
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:

        # 前缀和
        n = len(nums)
        presum_dct = {0:1}

        presum = 0
        ans = 0
        for i in range(n):
            presum += nums[i]

            if presum - k in presum_dct:
                ans += presum_dct[presum- k]
            presum_dct[presum] = presum_dct.get(presum, 0) + 1
        
        return ans 

```

date:20250728
### 15. 爬楼梯

Problem:假设你正在爬楼梯。需要 n 阶你才能到达楼顶。每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

Think: 动态规划： dp[i] = dp[i-1] + dp[i-2]

Solution:
```py   
class Solution:
    def climbStairs(self, n: int) -> int:

        dp = [0 for _ in range(n+1)]
        dp[0] = 1
        dp[1] = 1

        for i in range(2, n+1):
            dp[i] = dp[i-1] + dp[i-2]

        return dp[n]
```


### 16. 环形链表 II

Problem: 给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

Think: 使用快慢指针，若能相遇则有环;相遇后从初始位置和相遇位置分别开始遍历，相遇则为入环点

Solution:
```py  
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:

        slow = head
        fast = head
        while fast != None:
            slow = slow.next

            if fast.next == None:
                return None
            fast = fast.next.next

            if slow == fast:
                cur = head
                while cur != slow:
                    cur = cur.next
                    slow = slow.next
                return cur
        return None
```

### 17. 最大子数组和

Problem: 给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

Think: 动态规划： 子数组和 presum = max(presum + num, num)

Solution:
```py   
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # pre = max(pre + nums[i], nums[i])

        presum = 0
        ans = nums[0]
        for num in nums:
            presum = max(presum + num, num)
            ans = max(presum, ans)
        return ans
```

date: 20250729

### 18. 二叉树的最大深度

Problem: 给定一个二叉树 root ，返回其最大深度。二叉树的 最大深度 是指从根节点到最远叶子节点的最长路径上的节点数。

Think:递归求解，二叉树最大深度=MAX(左子树最大深度，右子树最大深度) + 1

Solution:
```py   
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:

        if root == None:
            return 0

        else:
            left = self.maxDepth(root.left)
            right = self.maxDepth(root.right)
            return max(left, right) + 1
```

### 19. 数组中的第K个最大元素

Problem: 给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。你必须设计并实现时间复杂度为 O(n) 的算法解决此问题。

Think: 先排序，再取第k大

Solution:
```py   
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        nums.sort(reverse=True)
        return nums[k-1]

```
date:20250730

### 20. 二叉树的中序遍历

Problem: 给定一个二叉树的根节点 root ，返回 它的 中序 遍历 。

Think: 先遍历左子树，先访问节点，再右子树


Solution:
```py   
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        ans = []
        def inorder(root):
            if root == None:
                return 

            inorder(root.left)
            ans.append(root.val)
            inorder(root.right)
        
        inorder(root)
        return ans
```

Solution2:
```py   
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        ans = []
        st = []
        while len(st) > 0 or root:

            # 左子树
            while root:
                st.append(root)
                root = root.left
            
            root = st.pop()
            ans.append(root.val)
            root = root.right
        return ans
```

### 21. 三数之和

Problem: 给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请你返回所有和为 0 且不重复的三元组。

think: 暴力三层循环； 先排序，再外层for + 双指针（注意去重）

Solution:

```py
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

```

### 22. 二叉树的直径

Problem: 给你一棵二叉树的根节点，返回该树的 直径 。二叉树的 直径 是指树中任意两个节点之间最长路径的 长度 。这条路径可能经过也可能不经过根节点 root 。

think: 求直径（即求路径长度的最大值）等效于求路径经过节点数的最大值减一。当前树的最大直径 = 左子树最大深度 + 右子树最大深度 + 1

Solution:

```py
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        
        self.ans = 0

        def depth(root):
            if root == None:
                return 0
        
            left = depth(root.left)
            right = depth(root.right)

            self.ans = max(self.ans, left + right + 1)

            return max(left, right) + 1
        
        depth(root)

        return self.ans - 1

```

### 23. 有效的括号

Problem: 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。

think: 使用栈保存左边括号，碰到右边出栈，否则入栈，最后栈空则满足

Solution:

```py
class Solution:
    def isValid(self, s: str) -> bool:

        st = []

        dct = {
            '(' : ')',
            '[' : ']',
            '{' : '}'
        }

        for char in s:
            if char in dct:
                st.append(char)  
            else:
                if len(st) > 0:
                    if char == dct[st[-1]]:
                        st.pop()
                    else:
                        return False
                else:
                    return False

        return False if len(st) > 0 else True
```

date:20250803

### 24.  翻转二叉树

Problem: 给你一棵二叉树的根节点 root ，翻转这棵二叉树，并返回其根节点。

think: 左右子树（翻转后的）互换 ，最后返回root

Solution:

```py
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:

        if root == None:
            return 
        
        left = self.invertTree(root.left)
        right = self.invertTree(root.right)

        root.left = right
        root.right = left

        return root
```

### 25. 二叉树的层序遍历

Problem: 给你二叉树的根节点 root ，返回其节点值的 层序遍历 。 （即逐层地，从左到右访问所有节点）。

think: 依次遍历，使用队列保存, python 中使用双端队列来保存 

左边操作： popleft, appendleft
右边操作： pop, append

Solution:

```py
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:

        if root is None:
            return []

        q = deque([root])
        ans = []
        while q:
            ret = []
            for _ in range(len(q)):
                root = q.popleft()
                ret.append(root.val)
                
                if root.left:
                    q.append(root.left)
                
                if root.right:
                    q.append(root.right)
            ans.append(ret)
        return ans
```

