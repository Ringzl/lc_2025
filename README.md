# lc_2025


## 基本问题


## 排序

1. 冒泡排序、选择排序、插入排序

2. 谢尔排序、堆排序、合并排序

3. 快速排序、基数排序

### 分治

1. 二分查找： 在排序数组中查找元素的第一个和最后一个位置

2. 分治


### 树

1. 二叉数的遍历（递归、迭代）

2. 二叉数的构造


### 回溯

1. 排列

2. 组合

### 图论

1. dfs、bfs

2. 无环图

3. 最短路径

### 动态规划

1. 01 背包

2. 完全背包




## LeetCode Hot 100

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

### 26. 对称二叉树

Problem: 给你一个二叉树的根节点 root ， 检查它是否轴对称。

think: 
1. 根节点左右孩子相同，左孩子的左孩子 == 右孩子的右孩子 左孩子的右孩子 = 右孩子的左孩子
2. 使用两个指针遍历，p指针、q指针分别遍历对应位置

Solution:

```py
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:

        def check(p, q):
            if p == None and q == None:
                return True
            if p== None or q == None:
                return False
            
            check_lr = check(p.left, q.right)
            check_rl = check(p.right, q.left)
            
            if p.val == q.val and check_lr and check_rl:
                return True
            else:
                return False

        return check(root.left, root.right)
```

### 27. 验证二叉搜索树

Problem: 给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。

think: 中序遍历是有序数组

Solution:

```py
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        
        lst = []
        def inorder(root):
            if root ==None:
                return 

            inorder(root.left)
            lst.append(root.val)
            inorder(root.right)

        inorder(root)
        for i in range(len(lst)-1):
            if lst[i] >= lst[i+1]:
                return False

        return True

```


### 28. 全排列

Problem: 给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。

think: 
* 终止条件： 长度为nums大小
* 遍历时标记是否访问过

Solution:

```py
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        path = []

        used = [False for _ in range(len(nums))]

        def backtrack(nums, used):

            if len(path) == len(nums):
                res.append(path.copy())
                return

            for i in range(len(nums)):
                if used[i] == True:
                    continue
                used[i] = True
                path.append(nums[i])
                backtrack(nums, used)
                path.pop()
                used[i] = False
            
        backtrack(nums, used)
        return res
```

### 29. 环形链表

Problem: 给你一个链表的头节点 head ，判断链表中是否有环。

think: 使用快慢指针，若能相遇，则有环

Solution:

```py
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:

        slow = head
        fast = head

        while fast != None:
            slow = slow.next

            if fast.next == None:
                return False

            fast = fast.next.next

            if slow == fast:
                return True
```

### 30. 子集

Problem: 给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。

think: 使用idnex作为参数，防止重复取值 for i in range(index, n): backtrack(i+1)

Solution:

```py

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:

        n = len(nums)
        ans = []
        path = []

        def backtrack(index):
            ans.append(path.copy())

            for i in range(index, n):
                path.append(nums[i])
                backtrack(i+1)
                path.pop()

        backtrack(0)
        return ans

```


### 31. 组合总和

Problem: 给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。

think: 终止条件 if sum(path) >= target， 无顺序防止重复 for i in range(index,n)， 可以重复选取 backtrack(i)

Solution:

```py
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:

        ans = []

        path = []
        n = len(candidates)
        def backtrack(index):
            if sum(path) >= target:
                if sum(path) == target:
                    ans.append(path.copy())
                return 
            
            for i in range(index,n):
                path.append(candidates[i])
                backtrack(i)
                path.pop()

        backtrack(0)

        return ans

```

date: 20250806

### 32. 合并两个有序链表

Problem: 将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

think: p,q 分别遍历两个链表，对比节点值大小，小的追加到新链表后面

Solution:

```py
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:

        list3 = ListNode()
        l3 = list3
        while list1 != None and list2 != None:
            if list1.val <= list2.val:
                new_val = list1.val
                list1 = list1.next
            else:
                new_val = list2.val
                list2 = list2.next


            node = ListNode(new_val)
            l3.next = node
            l3 = node
        
        while list1 != None:
            node = ListNode(list1.val)
            l3.next = node
            l3 = node
            list1 = list1.next

        while list2 != None:
            node = ListNode(list2.val)
            l3.next = node
            l3 = node
            list2 = list2.next

        return list3.next
```

### 33. 将有序数组转换为二叉搜索树

Problem: 给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 平衡 二叉搜索树。

think: 二叉搜索树中序遍历拿到有序数组，平衡：高度相差不超过1，树不唯一； 中序遍历，总是选择中间位置及左边的数字作为根节点

Solution:

```py
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:

        def helper(nums, left, right):

            if left > right:
                return 

            # 中间位置/左边作为根节点 向下取整
            mid = left + (right - left) // 2

            root = TreeNode(nums[mid])
            root.left = helper(nums, left, mid - 1)
            root.right = helper(nums, mid + 1, right)

            return root

        return helper(nums, 0, len(nums) - 1)
```

### 34. 岛屿数量

Problem: 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

think: 每发现一个陆地格子，进行dfs/bfs搜索，并标记访问过的格子


Solution:

```py
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        
        m, n = len(grid), len(grid[0])
        is_visited = [[False for _ in range(n)] for _ in range(m)]


        dirs = [
            [0,1], [0,-1], [1,0], [-1,0]
        ]



        def dfs(i, j):
            for d in dirs:
                ni = i + d[0]
                nj = j + d[1]
                if ni < 0 or ni >= m or nj < 0 or nj >= n or is_visited[ni][nj]:
                    continue
                
                if grid[ni][nj] == '1':
                    is_visited[ni][nj] = True
                    dfs(ni, nj)


        def bfs(i, j):
            q = deque()
            is_visited[i][j] = True
            q.append((i,j))

            while len(q) > 0:
                x, y = q.popleft()
                for d in dirs:
                    nx = x + d[0]
                    ny = y + d[1]

                    if nx < 0 or nx >= m or ny < 0 or ny >= n or is_visited[nx][ny]:
                        continue

                    if grid[nx][ny] == '1':
                        is_visited[nx][ny] = True
                        q.append((nx, ny))
            
        ans = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1' and not is_visited[i][j]:
                    # dfs(i,j)
                    bfs(i,j)
                    ans += 1

        return ans

```

date:20250810
### 35. 矩阵置零

Problem: 给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。

think: 遍历所有元素，保存0所在行、列

Solution:

```py
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        m, n = len(matrix), len(matrix[0])
        rows = set()
        cols = set()
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    rows.add(i)
                    cols.add(j)
        
        for row in rows:
            matrix[row][:] = [0 for _ in range(n)]
        
        for i in range(m):
            for col in cols:
                matrix[i][col] = 0

        return matrix
```

### 36.  电话号码的字母组合

Problem:给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

think: 先构建映射，使用index纵向形成解，横向遍历每个数字可取的字母，注意回溯 backtrack(index+1)


Solution: 

```py
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:

        nc_dct = {
            '2': 'abc',
            '3': 'def', 
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv', 
            '9': 'wxyz'
        }

        if digits == "":
            return []
        
        ans = []
        self.path = ""
        n = len(digits)
        
        def backtrack(index):
            if index == n:
                ans.append(self.path)
            else:
                for c in nc_dct[digits[index]]:
                    self.path += c
                    backtrack(index+1)
                    self.path = self.path[:-1]

        backtrack(0)
        return ans
```

### 37. 搜索二维矩阵

Problem:给你一个满足下述两条属性的 m x n 整数矩阵：每行中的整数从左到右按非严格递增顺序排列。
每行的第一个整数大于前一行的最后一个整数。给你一个整数 target ，如果 target 在矩阵中，返回 true ；否则，返回 false 。

think: 二维当一维看，二分法

Solution:

```py
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        
        m, n = len(matrix), len(matrix[0])
        left = 0 
        right = m * n -1

        while left <= right:

            mid = left + (right - left) // 2
            x = mid // n
            y = mid % n

            if matrix[x][y] == target:
                return True

            elif matrix[x][y] < target:
                left = mid + 1
            else:
                right = mid - 1

        return False
```

### 38. 买卖股票的最佳时机

Problem: 给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。

think: 买入、卖出时机 i、j, ans = max(ans, prices[j] - min_price), 

Solution:

```py
class Solution:
    def maxProfit(self, prices: List[int]) -> int:

        ans = 0
        n = len(prices)

        min_price = float('inf')
        for i in range(n):
            if prices[i] < min_price:
                min_price = prices[i]
            else:
                ans = max(ans, prices[i] - min_price )

        return ans
```

### 39. 括号生成

Problem: 数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。

think: 回溯，停止条件 左右括号数相同保存解 left == right， left < right 或 left > n return

Solution:

```py
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        
        ans = []
        self.path = ""
        def backtrack(left, right):
            if (left == n and right == n):
                ans.append(self.path)
                return
            
            if left < right or left > n:
                return 
            
            self.path += '('
            backtrack(left + 1, right)
            self.path = self.path[:-1]

            self.path += ')'
            backtrack(left, right+1)
            self.path = self.path[:-1]

        backtrack(0,0)

        return ans
```

date:20250812

### 40. 腐烂的橘子

Problem:在给定的 m x n 网格 grid 中，每个单元格可以有以下三个值之一：
值 0 代表空单元格；
值 1 代表新鲜橘子；
值 2 代表腐烂的橘子。
每分钟，腐烂的橘子 周围 4 个方向上相邻 的新鲜橘子都会腐烂。
返回 直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 -1 。

think: 直接 bfs 层序遍历，标记腐烂，最后返回层数

Solution:

```py
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:

        m, n = len(grid), len(grid[0])
        cnt = 0
        q = deque()
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    cnt += 1
                elif grid[i][j] == 2:
                    q.appendleft([i,j])
        if cnt == 0:
            return 0

        ans = 0

        dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]]

        while cnt > 0 and len(q) > 0:
            size = len(q)

            for i in range(size):
                x, y = q.pop()

                for d in dirs:
                    nx, ny = x + d[0], y + d[1]

                    # 未超出边界且新鲜
                    if nx >= 0 and nx < m and ny >= 0 and ny < n and  grid[nx][ny] == 1:
                        cnt -= 1
                        grid[nx][ny] = 2
                        q.appendleft([nx,ny])
            ans += 1
        
        if cnt > 0:
            return -1
        else:
            return ans
```

### 41. 找到字符串中所有字母异位词

Problem: 给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

think: 
1. 直接排序s和p, 遍历s对比p
2. 用两个列表表示每个字符出现次数
Solution:

```py
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:

        # p = sorted(p)
        # n = len(s)
        # np = len(p)

        # ans = []
        # for i in range(n-np+1):
        #     tmp = sorted(s[i:i+np])
        #     if tmp == p:
        #         ans.append(i)
        # return ans

        s_cnt = [0 for _ in range(26)]
        p_cnt = [0 for _ in range(26)]

        for i in range(len(p)):
            p_cnt[ord(p[i]) - ord('a')] += 1
        ans = []
        left = 0
        for right in range(len(s)):
            s_cnt[ord(s[right]) - ord('a')] += 1

            if right - left + 1 == len(p):
                if s_cnt == p_cnt:
                    ans.append(left)
                s_cnt[ord(s[left]) - ord('a')] -= 1
                left += 1
```

### 42. 杨辉三角

Problem: 给定一个非负整数 numRows，生成「杨辉三角」的前 numRows 行。

think:  第n行有n个数，分别为 1 a[n][i] = a[n-1][i-1] + a[n-1][i] 

Solution:

```py
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:

      if numRows == 0:
        return []
      else:
        ans = [[1]]

      for i in range(1, numRows):

        tmp = [1]
        for j in range(1, i):
          tmp.append(ans[i-1][j-1] + ans[i-1][j])

        tmp.append(1)

        ans.append(tmp)

      return ans
```

### 43. 打家劫舍

Problem: 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。

think: 
1. 不相邻，可以偷到最多的金额
2. 偷k, 总金额为前k-2间房屋 + 第k
3. 不偷k,总金额为前k-1间房屋 
dp[i] = max(dp[i-2]+nums[i], dp[i-1])


Solution:

```py
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        if n == 1:
            return nums[0]
        
        dp = [0 for _ in range(n)]

        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])

        for i in range(2, n):
            dp[i] = max(dp[i-2] + nums[i], dp[i-1])
        
        return dp[n-1]
```

### 44. 完全平方数

Problem: 给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。

think: dp[i] 表示最少需要的平方数表示i, 这些数小于等于 $\sqrt(i)$


Solution:

```py
class Solution:
    def numSquares(self, n: int) -> int:
        
        dp = [0 for _ in range(n+1)]
        j = 1
        for i in range(1, n+1):
            dp[i] = i # 最坏情况
            while j * j <= i:
                dp[i] = min(dp[i], dp[i-j*j] + 1)
                j+=1

        return dp[n]
```

### 45. 两数相加

Problem: 给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。请你将两个数相加，并以相同形式返回一个表示和的链表。

think: 
暴力解法： 将链表数字解析出来求和再新建链表添加

2: 将两个链表看成是相同长度的进行遍历，如果一个链表较短则在前面补 0, 每一位计算的同时需要考虑上一位的进位问题，而当前位计算结束后同样需要更新进位值

Solution:

```py
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        
        # s1 = ""
        # while l1 != None:
        #     s1 = str(l1.val) + s1
        #     l1 = l1.next
        
        # s2 = ""
        # while l2 != None:
        #     s2 = str(l2.val) + s2
        #     l2 = l2.next

        # s_sum = reversed(str(int(s1) + int(s2)))

        # res = ListNode()
        # p = res
        # for s in s_sum:
        #     p.next = ListNode(int(s))
        #     p = p.next
        
        # return res.next

        pre = ListNode(0)
        cur = pre
        carry = 0
        while (l1 != None or l2 != None):

            x = l1.val if l1 != None else 0
            y = l2.val if l2 != None else 0

            s = x + y + carry
            carry = s // 10
            s = s % 10
            cur.next = ListNode(s)

            cur = cur.next

            if l1 != None:
                l1 = l1.next
            if l2 != None:
                l2 = l2.next

        if carry != 0:
            cur.next = ListNode(carry)

        return pre.next
```

### 46. 螺旋矩阵

Problem: 给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。

think: 使用 is_visited 保存是否访问，转弯时顺时针旋转进入下一个方向

Solution:

```py
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:

        m, n = len(matrix), len(matrix[0])

        # # 顺序 右、下、左、上
        # dirs = [
        #     [0,1], [1,0], [0,-1], [-1,0]
        # ]

        # is_visited = [
        #     [False for _ in range(n)] for _ in range(m)
        # ]

        # total = m * n

        # row, col = 0,0
        # dir_index = 0 # 初始方向
        # ans = []
        # for i in range(total):
        #     ans.append(matrix[row][col])
        #     is_visited[row][col] = True

        #     next_row, next_col = row + dirs[dir_index][0], col + dirs[dir_index][1]

        #     if next_row < 0 or next_row >= m or next_col < 0 or next_col >= n or is_visited[next_row][next_col]:
        #         dir_index = (dir_index + 1) % 4

        #     row += dirs[dir_index][0]
        #     col += dirs[dir_index][1]

        # 分层模拟
        left, right, top, bottom = 0, n-1, 0, m-1

        ans = []
        while left <= right and top <= bottom:
            
            # 从左到右
            for i in range(left, right+1):
                ans.append(matrix[top][i])
            top += 1

            # 从上到下
            for i in range(top, bottom+1):
                ans.append(matrix[i][right])
            right -= 1

            # 从右到左
            if top <= bottom:
                for i in range(right, left-1, -1):
                    ans.append(matrix[bottom][i])
            bottom -= 1
            
            # 从下到上
            if left <= right:
                for i in range(bottom, top-1, -1):
                    ans.append(matrix[i][left])
            left += 1

        return ans
```

### 47. 二叉搜索树中第 K 小的元素

Problem: 定一个二叉搜索树的根节点 root ，和一个整数 k ，请你设计一个算法查找其中第 k 小的元素（从 1 开始计数）。


think: 中序遍历得到有序数组，再取第k个数

Solution:

```py
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:

        # nums = []

        # def inorder(root):
        #     if root == None:
        #         return

        #     inorder(root.left)
        #     nums.append(root.val)
        #     inorder(root.right)

        
        # inorder(root)

        # return nums[k-1]

        st = []

        while root or st:

            while root:
                st.append(root)
                root = root.left
            
            root = st.pop()
            k -= 1
            if k == 0:
                return root.val
            
            root = root.right 
```

### 48. 删除链表的倒数第 N 个结点

Problem: 给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

think: 双指针间隔n, 右边指针到表尾时左边指针即为倒数第n个节点

Solution:

```py
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:

        left, right = head, head

        for i in range(n):
            right = right.next

        pre = None
        while right != None:
            pre = left
            left = left.next
            right = right.next

        if pre != None:
            pre.next = left.next
            return head
        elif left != None:
            return left.next
        else:
            return None
```

### 49. 在排序数组中查找元素的第一个和最后一个位置

Problem: 给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。如果数组中不存在目标值 target，返回 [-1, -1]。


think: 
先找第一个等于: 等于时往左找
再找最后一个等于：等于时往右找

Solution:

```py
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:

        n = len(nums)

        ans = [-1, -1]

        # 先找第一个等于
        left, right = 0, n-1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                ans[0] = mid
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        # 再找最后一个等于
        left, right = 0, n-1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                ans[1] = mid
                left = mid + 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return ans
```

### 50. 二叉树的右视图

Problem: 给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。

think: 层序遍历，保存每一层最后一个数

Solution:

```py
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        
        q = deque()
        if root == None:
            return []
        q.appendleft(root)
        ans = []
        while len(q) > 0:
            size = len(q)

            for i in range(size):
                node = q.pop()

                if i == size-1:
                    ans.append(node.val)
                if node.left:
                    q.appendleft(node.left)
                if node.right:
                    q.appendleft(node.right)
        return ans
```

### 51. 零钱兑换

Problem: 给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。你可以认为每种硬币的数量是无限的。

think: 
dp[i] 表示凑成总金额i所需要的硬币个数
dp[i] = min(dp[i-coins[j]] + 1, dp[i])

Solution:

```py
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        
        n = len(coins)

        dp = [float('inf') for _ in range(amount+1)]
        dp[0] = 0
        for i in range(amount+1):
            for j in range(n):
                if i - coins[j] >= 0:
                    dp[i] = min(dp[i], dp[i-coins[j]] + 1)
        
        return dp[amount] if dp[amount] != float('inf') else - 1
```

### 52. 只出现一次的数字 

problem: 给你一个 非空 整数数组 nums ，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

think: 
1. 使用字典统计次数
2. 数组中的全部元素的异或运算结果即为数组中只出现一次的数字

Solution:

```py
class Solution:
    def singleNumber(self, nums: List[int]) -> int:

        # num_cnt = {}
        # n = len(nums)
        # for i in range(n):
        #     num_cnt[nums[i]] = num_cnt.get(nums[i], 0) + 1
        
        # for num in num_cnt:
        #     if num_cnt[num] == 1:
        #         return num
        ans = 0
        for num in nums:
            ans ^= num
        return ans
```

### 53. 不同路径

problem: 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。问总共有多少条不同的路径？


think: dp[i][j] 表示到达 (i,j)的路径数量，dp[i][j] = dp[i-1][j] + dp[i][j-1]


Solution:

```py
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:

        dp = [[0 for _ in range(n)] for _ in range(m)]

        for j in range(n):
            dp[0][j] = 1
        for i in range(m):
            dp[i][0] = 1

        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]

        return dp[m-1][n-1]
```

### 54. 最小栈

problem: 设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。

实现 MinStack 类:
MinStack() 初始化堆栈对象。
void push(int val) 将元素val推入堆栈。
void pop() 删除堆栈顶部的元素。
int top() 获取堆栈顶部的元素。
int getMin() 获取堆栈中的最小元素。

think: 使用当前最小值栈保存最小值记录

Solution:

```py

class MinStack:

    def __init__(self):
        self.st = []
        self.min_st = [math.inf]

    def push(self, val: int) -> None:
        self.st.append(val)
        self.min_st.append(min(self.min_st[-1], val))

    def pop(self) -> None:
        self.st.pop()
        self.min_st.pop()
        

    def top(self) -> int:
        return self.st[-1]
        

    def getMin(self) -> int:
        return self.min_st[-1]
        

```


### 55. 跳跃游戏

problem: 给你一个非负整数数组 nums ，你最初位于数组的 第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。判断你是否能够到达最后一个下标，如果可以，返回 true ；否则，返回 false 。

think: 位置可达 i<=max_pos, 更新最远可达 max_pos = max(max_pos, i + nums[i])

Solution:

```py
class Solution:
    def canJump(self, nums: List[int]) -> bool:

        max_pos = 0
        n = len(nums)
        for i in range(n):

            if i <= max_pos:
                max_pos = max(max_pos, i + nums[i])

                if max_pos >= n-1:
                    return True

        return False
```

### 56. 相交链表

problem: 给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null 。

think: 
1. 用set保存访问过的节点，判断交点
2. 如果两个A,B链表走相同的距离会重合

Solution:

```py
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:

        # A = headA
        # B = headB

        # while A != B:
        #     A = A.next if A else headB
        #     B = B.next if B else headA

        # return A

        m = 0
        n = 0
        p, q = headA, headB
        while p != None:
            p = p.next
            m += 1
        while q != None:
            q = q.next
            n += 1

        p, q = headA, headB
        if m <= n:
            for i in range(n-m):
                q = q.next
        else:
            for i in range(m-n):
                p = p.next

        while (q != None) and (p != None):
            if p == q:
                return p
            p = p.next
            q = q.next

        return None
```

### 57. 乘积最大子数组

problem: 给你一个整数数组 nums ，请你找出数组中乘积最大的非空连续 子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。测试用例的答案是一个 32-位 整数。


think: 动态规划，dp[i] 为前i个数 乘积最大的非空连续 子数组的乘积

nums[i] >= dp[i-1] >= 0, dp[i] = nums[i]
dp[i-1] < 0 and nums[i] < 0, dp[i] = mindp[i-1] * nums[i]
nums[i] > 0, d[i] = dp[i-1] * nums[i]


Solution:

```py
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
```

### 58. 二叉树展开为链表

problem: 给你二叉树的根结点 root ，请你将它展开为一个单链表：展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。展开后的单链表应该与二叉树 先序遍历 顺序相同。

think: 
1. 递归先序遍历改树
2. 将左子树插入到右子树的地方；将原来的右子树接到左子树的最右边节点；考虑新的右子树的根节点，一直重复上边的过程，直到新的右子树为 null

Solution:

```py
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        
        # val_lst = []
        # def preorder(root):
        #     if root == None:
        #         return 
        #     val_lst.append(root)
        #     preorder(root.left)
        #     preorder(root.right)
        # preorder(root) 
       
        # for i in range(1, len(val_lst)):
        #     prev, cur = val_lst[i-1], val_lst[i]
        #     prev.left = None
        #     prev.right = cur

        while root:
            # 左子树为 null，直接考虑下一个节点
            if root.left == None:
                root = root.right
            else:
                # 找左子树最右边的节点
                pre = root.left
                while pre.right:
                    pre = pre.right
                
                # 将原来的右子树接到左子树的最右边节点
                pre.right = root.right

                # 将左子树插入到右子树的地方
                root.right = root.left
                root.left = None

                root = root.right
```

### 59. 最小路径和

problem: 给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。说明：每次只能向下或者向右移动一步。

think: 

dp[i][j] 表示从左上角到 （i,j）路径数字最小总和

dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + nums[i][j]

Solution:

```py
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:

        m = len(grid)
        n = len(grid[0])

        import math
        dp = [[math.inf for _ in range(n)] for _ in range(m)]

        dp[0][0] = grid[0][0]
        for j in range(1, n):
            dp[0][j] = dp[0][j-1] + grid[0][j]
        
        for i in range(1, m):
            dp[i][0] = dp[i-1][0] + grid[i][0]

        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        
        return dp[m-1][n-1]
```

### 60. 最长回文子串

problem: 给你一个字符串 s，找到 s 中最长的 回文 子串。

think: if s[i] == s[j] and dp[i+1][j-1] == 1: dp[i][j] = 1
  
dp[i][j] 表示i开始j结束是否为回文串
dp[i][i]=1; //单个字符是回文串
dp[i][i+1]=1 if s[i]=s[i+1]; //连续两个相同字符是回文串

Solution:

```py
class Solution:
    def longestPalindrome(self, s: str) -> str:
        
        n = len(s)
        dp = [[0 for _ in range(n)] for _ in range(n)]

        if n == 0 or n == 1:
            return s

        max_l = 1
        start = 0 
        # 初始化
        for i in range(n):
            dp[i][i] = 1
            if i < n-1 and s[i] == s[i+1]:
                dp[i][i+1] = 1
                start = i
                max_l = 2
                

        for l in range(3, n+1):

            for i in range(n-l+1):
                j = i + l -1

                if s[i] == s[j] and dp[i+1][j-1] == 1:
                    dp[i][j] = 1
                    start = i
                    max_l = l
        
        return s[start:start+max_l]
```

### 61. 轮转数组

problem: 给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。

think: i 轮转 k 个位置后为 (i + k) % n, 重新赋值


Solution:

```py
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        # n = len(nums)

        # ans = [0 for _ in range(n)]
        # for i in range(n):
        #     ans[(i+k)%n] = nums[i]
        
        # for i in range(n):
        #     nums[i] = ans[i]

        # 翻转
        n = len(nums)
        k = k % n
        if k != 0:
            tmp = nums[-k:]
            nums[k-n:] = nums[0:n-k]
            nums[0:k] = tmp
        
```

[20250830]

### 62. 除自身以外数组的乘积

problem: 给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积 。

think: 先分别计算数组元素左侧乘积数组L，和右侧乘积数组R, ans[i] = L[i] * R[i]


Solution:

```py
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        
        n = len(nums)

        # ans = []
        # for i in range(n):
        #     tmp = 1
        #     for j in range(n):
        #         if j != i:
        #             tmp *= nums[j]
        #     ans.append(tmp)
        # return ans

        L = [1 for _ in range(n)] # 左侧乘积
        for i in range(1, n):
            L[i] = nums[i-1] * L[i-1]

        R = [1 for _ in range(n)] # 右侧乘积
        for i in range(n-2, -1, -1):
            R[i] = nums[i+1] * R[i+1]
        
        ans = []
        for i in range(n):
            ans.append(L[i] * R[i])
        
        return ans
```

### 63. 旋转图像

problem: 给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。

think: 旋转 90 度==> 先上下翻转，再对称翻转


Solution:

```py
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """

        n = len(matrix)

        for i in range(n // 2):
            tmp = matrix[i]
            matrix[i] =  matrix[n-i-1]
            matrix[n-i-1] = tmp
        
        for i in range(n):
            for j in range(n):
                if i < j:
                    matrix[i][j], matrix[j][i]= matrix[j][i], matrix[i][j]
        
        return matrix
```

### 64. 搜索二维矩阵 II

problem: 编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：
每行的元素从左到右升序排列。
每列的元素从上到下升序排列。

think: 
1. 对每一行进行二分搜索
2. 从根（右上角）开始搜索


Solution:

```py
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:

        m, n = len(matrix), len(matrix[0])

        # for i in range(m):

        #     l, r = 0, n-1
        #     while l <= r:
        #         mid = l + (r-l)//2

        #         if matrix[i][mid] == target:
        #             return True
        #         elif matrix[i][mid] > target:
        #             r = mid - 1
        #         else:
        #             l = mid + 1

        # return False

        r, c = 0, n-1
        while r < m and c >= 0:

            if matrix[r][c] == target:
                return True
            elif matrix[r][c] < target:
                r += 1
            else:
                c -= 1
        
        return False
```

### 65. 两两交换链表中的节点

problem: 给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。

think: 

使用哑变量节点保存前一个节点 pre
交换 node1 和 node2

pre.next = node2
node1.next = node2.next
node2.next = node1
pre = node1

Solution:

```py
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:


        dummy = ListNode()
        dummy.next = head
        pre = dummy
        while pre.next != None and pre.next.next != None:
            node1 = pre.next
            node2 = pre.next.next

            pre.next = node2
            node1.next = node2.next
            node2.next = node1
            pre = node1

        return dummy.next
```

### 66. 随机链表的复制

problem: 给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。

think: 因为随机指针的存在，当我们拷贝节点时，「当前节点的随机指针指向的节点」可能还没创建


Solution:

```py
class Solution:
    def __init__(self):
        self.cacheNode = dict()
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':

        if head == None:
            return 

        if head not in self.cacheNode:
            headNew = Node(head.val)
            self.cacheNode[head] = headNew

            headNew.next = self.copyRandomList(head.next)
            headNew.random = self.copyRandomList(head.random)

        return self.cacheNode[head]
        
```

[20250831]

### 67. 排序链表

problem: 给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。


think: 先取出值到列表，再排序，最后放回

Solution:

```py

class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        
        p = head
        nums = []
        while p:
            nums.append(p.val)
            p = p.next
        
        nums.sort()

        p = head
        i = 0
        while p:
            p.val = nums[i]
            p = p.next
            i += 1

        return head 
```

### 68. LRU 缓存

problem: 请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
实现 LRUCache 类：
LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。
函数 get 和 put 必须以 O(1) 的平均时间复杂度运行


think: 双向链表 + hash表
key 存在，变更其数据值 value；不存在，则向缓存中插入该组 key-value；
get  O(1) hash表 （key-value），使用过后移到链表头部
put O(1) 链表，移到链表头部，超容量删除表尾


Solution:

```py
class DLinkedNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:

    def __init__(self, capacity: int):
        self.cache = dict()

        self.head = DLinkedNode()
        self.tail = DLinkedNode()

        self.head.next = self.tail
        self.tail.prev = self.head
        self.capacity = capacity
        self.size = 0
        
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        
        # key 存在，移到头部
        node = self.cache[key]
        self.moveToHead(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key not in self.cache:
            # key 不存在，创建一个新节点
            node = DLinkedNode(key, value)

            # 添进hash表,添加到表头
            self.cache[key] = node
            self.addToHead(node)
            self.size += 1

            if self.size > self.capacity:
                # 删除表尾
                removed = self.removeTail()
                self.cache.pop(removed.key)
                self.size -= 1
        else:
            # key 存在，修改value值， 移到头部
            node = self.cache[key]
            node.value = value
            self.moveToHead(node)
    
    def addToHead(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def moveToHead(self, node):
        self.removeNode(node)
        self.addToHead(node)

    def removeNode(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def removeTail(self):
        node = self.tail.prev
        self.removeNode(node)
        return node

class LRUCache(collections.OrderedDict):

    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity

        
    def get(self, key: int) -> int:
        if key not in self:
            return -1
        
        self.move_to_end(key)
        return self[key]

    def put(self, key: int, value: int) -> None:
        if key in self:
            self.move_to_end(key)
        self[key] = value
        if len(self) > self.capacity:
            self.popitem(last=False)
```

20250901

### 69. 从前序与中序遍历序列构造二叉树

problem: 给定两个整数数组 preorder 和 inorder ，其中 preorder 是二叉树的先序遍历， inorder 是同一棵树的中序遍历，请构造二叉树并返回其根节点。

think: 
前序列表第一个值为root, 在中序遍历找到root的位置，左边为左子树，右边为右子树； 
递归构建：
- 从前序遍历中取出第一个元素作为当前根节点。
- 在中序遍历中找到根节点的位置，以此划分左右子树的节点集合。
- 递归构建左右子树。

Solution:

```py
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        
        # 中序元素位置
        inorder_map = dict()
        for i in range(len(inorder)):
            inorder_map[inorder[i]] = i
        

        # 建树
        # pStart, pEnd: 前序遍历的起始和结束位置
        # iStart, iEnd: 中序遍历的起始和结束位置
        def build(pStart, pEnd, iStart, iEnd):
            if pStart > pEnd or iStart > iEnd:
                return None
            
            # 前序遍历的第一个元素是根节点
            root_val = preorder[pStart]
            root = TreeNode(root_val)

            # 在中序遍历中找到根节点的位置
            root_pos = inorder_map[root_val]

            # 计算左子树的节点数量
            left_num = root_pos - iStart
            root.left = build(pStart+1, pStart+left_num, iStart, root_pos-1)
            root.right = build(pStart+left_num+1, pEnd, root_pos+1, iEnd)
            return root
        
        return build(0, len(preorder)-1, 0, len(inorder)-1)
```


### 70. 路径总和 III

problem: 给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。路径 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

think: 
路径方向必须是向下： 穷举所有的可能，我们访问每一个节点 node，检测以 node 为起始节点且向下延深的路径有多少种

Solution:

```py
class Solution:
    def rootSum(self, root, targetSum):
        if root == None:
            return 0
        
        ret = 0
        if root.val == targetSum:
            ret += 1
        
        ret += self.rootSum(root.left, targetSum - root.val)
        ret += self.rootSum(root.right, targetSum - root.val)

        return ret

    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        if root == None:
            return 0

        ret = self.rootSum(root, targetSum)
        ret += self.pathSum(root.left, targetSum)
        ret += self.pathSum(root.right, targetSum)

        return ret

# 搜索完以某个节点为根的左右子树之后，应当回溯地将路径总和从哈希表中删除，防止统计到跨越两个方向的路径。
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:

        prefix = Counter()
        prefix[0] = 1
        ans = 0

        def dfs(root, Sum):
            if root == None:
                return 
            Sum += root.val
            nonlocal ans
            ans += prefix[Sum - targetSum]
            prefix[Sum] += 1
            dfs(root.left, Sum)
            dfs(root.right, Sum)
            prefix[Sum] -= 1
        
        dfs(root, 0)
        return ans        
```

### 71. 二叉树的最近公共祖先

problem: 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先（满足 x 是 p、q 的祖先且 x 的深度尽可能大）。

think: 
条件：
1. 左子树和右子树均包含 p 节点或 q 节点 
2. x 恰好是 p 节点或 q 节点且它的左子树或右子树有一个包含了另一个节点

Solution:

```py
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        

        if root == None or root == p or root == q:
            return root

        # 左子树找p或q
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        # 左右都找到
        if left and right:
            return root
        
        # 只找到一个
        return left if left else right
```

[20250903]

### 72. 课程表

problem: 你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。

think: 拓扑排序： 图 G 中存在环，图 G 不存在拓扑排序。

拓扑排序中最前面的节点没有入边；
一个节点加入答案中后，我们就可以移除它的所有出边，代表着它的相邻节点少了一门先修课程的要求；
某个相邻节点变成了「没有任何入边的节点」，那么就代表着这门课可以开始学习
不断地将没有入边的节点加入答案，直到答案中包含所有的节点 / 或者不存在没有入边的节点

Solution:

```py
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        
        # 建图
        edges = [[] for _ in range(numCourses)]
        indegree = [0 for _ in range(numCourses)]

        for info in prerequisites:
            edges[info[1]].append(info[0])
            indegree[info[0]] += 1

        q = deque()
        for i in range(numCourses):
            if indegree[i] == 0:
                q.appendleft(i)
        
        visited = 0
        while q:
            visited += 1
            u = q.pop()

            for v in edges[u]:
                indegree[v] -= 1
                if indegree[v] == 0:
                    q.appendleft(v)
        
        return visited == numCourses
```

### 73. 实现 Trie (前缀树)


problem: Trie（发音类似 "try"）或者说 前缀树 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补全和拼写检查。\
Trie() 初始化前缀树对象。
void insert(String word) 向前缀树中插入字符串 word 。
boolean search(String word) 如果字符串 word 在前缀树中，返回 true（即，在检索之前已经插入）；否则，返回 false 。
boolean startsWith(String prefix) 如果之前已经插入的字符串 word 的前缀之一为 prefix ，返回 true ；否则，返回 false 。

think: 

判断给定的前缀是否是加入的字符串的前缀 -> 枚举给定前缀的每一位，查看字符串中是否存在字符的当前位是这个位；如果存在，就在这些字符串中去查找下一位。 -> 根据前缀对字符串进行分组

相同前缀只存储一次【从根节点出发到任一个节点都是一个前缀】

可以使用长度为 26 的列表来存储当前节点对应出现过的字符的子节点

Solution:

```py
class Node:
    def __init__(self):
        self.children = [None] * 26 
        self.isEnd = False 

class Trie:

    def __init__(self):
        self.root = Node()
        
    def insert(self, word: str) -> None:
        # 从根节点开始构造word对应的路径节点
        node = self.root

        for c in word:
            char_id = ord(c) - ord('a')
            if not node.children[char_id]:
                node.children[char_id] = Node()
            node = node.children[char_id]
        node.isEnd = True
        

    def search(self, word: str) -> bool:
        node = self.root
        for c in word:
            node = node.children[ord(c) - ord('a')]
            if not node:
                return False
        return node.isEnd

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for c in prefix:
            node = node.children[ord(c) - ord('a')]
            if not node:
                return False
        return True
```

### 74. 单词搜索

problem: 给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

think: 
1. 遍历二维数组 每一个位置都作为起点进行dfs，注意记录访问过的位置
2. 实现四个方向的dfs

Solution:

```py
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:


        def dfs(board, i, j, word, index, is_visited):

            if index == len(word):
                return True
            
            if i<0 or i>=len(board) or j<0 or j>=len(board[0]) or board[i][j] != word[index] or is_visited[i][j]:
                return False  
            
            is_visited[i][j] = True
            found = (
                dfs(board, i+1, j, word, index+1, is_visited) or 
                dfs(board, i-1, j, word, index+1, is_visited) or 
                dfs(board, i, j+1, word, index+1, is_visited) or 
                dfs(board, i, j-1, word, index+1, is_visited)
            )    
            is_visited[i][j] = False

            return found

        m, n = len(board), len(board[0])
        is_visited = [[False for _ in range(n)] for _ in range(m)]

        for i in range(m):
            for j in range(n):
                if dfs(board, i, j, word, 0, is_visited):
                    return True
        
        return False
```

20250906

### 75. 分割回文串

problem: 给你一个字符串 s，请你将 s 分割成一些 子串，使每个子串都是 回文串 。返回 s 所有可能的分割方案。


think: 
回溯 + dp
先判断子串 s[i,j] 是否为回文串： s[i+1,j-1] 为回文串 and s[i] == s[j]

is_huiwen[i][j] = (is_huiwen[i+1][j-1] and s[i] == s[j])

搜索到字符串的第 i 个字符，且 s[0:i] 已被分割，需要枚举下一个子串的右边界使得s[i,j]为回文串



Solution:

```py
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        n = len(s)
        is_huiwen = [
            [True for _ in range(n)] for _ in range(n)
        ]
        for i in range(n-1, -1, -1):
            for j in range(i+1, n):
                is_huiwen[i][j] = (is_huiwen[i+1][j-1] and s[i] == s[j])

        ret = []
        ans = []
        def dfs(s, i):
            if i == n:
                ret.append(ans.copy())
                return 

            for j in range(i, n):
                if is_huiwen[i][j]:
                    ans.append(s[i:j+1])
                    dfs(s, j+1)
                    ans.pop()
        
        dfs(s, 0)

        return ret
```

### 76. 字符串解码

problem: 给定一个经过编码的字符串，返回它解码后的字符串。编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。

think: 
使用栈维护：
解析出数字，读取数字存入栈
字母或'[' 直接入栈
']' 开始处理：
字母出栈，用数组保存，直到‘[’
后重复 st[-1] 次


Solution:

```py
class Solution:
    def decodeString(self, s: str) -> str:
        st = []

        i = 0
        while i < len(s):

            cur = s[i]

            if cur.isdigit():
                # 获取数字
                num = ''
                while s[i].isdigit():
                    num += s[i]
                    i += 1
                st.append(num)
            elif cur.isalpha() or cur == '[':
                st.append(s[i])
                i += 1
            else:
                i += 1
                sub = []
                while st[-1] != '[':
                    sub.append(st[-1])
                    st.pop()
                sub = sub[::-1]
                sub = ''.join(sub)
                
                # 左括号出栈
                st.pop()

                # 栈顶为次数
                rep = int(st[-1])
                st.pop()

                t = rep * sub
                st.append(t)
                
                
        return ''.join(st)

```

### 77. 搜索旋转排序数组

problem: 整数数组 nums 按升序排列，数组中的值 互不相同 。在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 向左旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 下标 3 上向左旋转后可能变为 [4,5,6,7,0,1,2] 。

think: 
1. 二分法确定有序部分
2. 在有序部分查找


Solution:

```py
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
```

### 78. 寻找旋转排序数组中的最小值
problem: 已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,2,4,5,6,7] 在变化后可能得到：
若旋转 4 次，则可以得到 [4,5,6,7,0,1,2]
若旋转 7 次，则可以得到 [0,1,2,4,5,6,7]
注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。

think: 
二分法查找
右侧有序： r = mid
否则： l = mid + 1

Solution:

```py
class Solution:
    def findMin(self, nums: List[int]) -> int:
        n = len(nums)
        l = 0
        r = n -1

        while l < r:
            mid = l + (r -l) // 2
            if nums[mid] < nums[r]:
                r = mid
            else:
                l = mid + 1
        
        return nums[l]
```

### 79. 前 K 个高频元素
problem: 给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。

think: 
先统计频率，再按照频次排序，取前k个元素

用大小为k的最小堆保存

Solution:

```py
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        
        n = len(nums)
        freq_dct = {}
        for i in range(n):
            freq_dct[nums[i]] = freq_dct.get(nums[i], 0) + 1
        
        # freq_set = sorted(freq_dct.items(), key = lambda x:x[1], reverse= True)
        # ans = []
        # for i in range(k):
        #     ans.append(freq_set[i][0])

        # 用大小为k的最小堆保存
        heap_k = []
        for key in freq_dct:
            heapq.heappush(heap_k, [cnt_dct[key], key])

            if len(heap_k) > k:
                heapq.heappop(heap_k)
        ans = []
        while heap_k:
            ans.append(heapq.heappop(heap_k)[1])

    
        return ans
```

### 80. 跳跃游戏 II
problem: 给定一个长度为 n 的 0 索引整数数组 nums。初始位置在下标 0。每个元素 nums[i] 表示从索引 i 向后跳转的最大长度。换句话说，如果你在索引 i 处，你可以跳转到任意 (i + j) 处：0 <= j <= nums[i] 且
i + j < n 返回到达 n - 1 的最小跳跃次数。测试用例保证可以到达 n - 1。


think: 

位置i最远可达： max_pos = max(max_pos, i + nums[i])

if i == end: 
    end = max_pos // 每次走最远
    ans += 1


Solution:

```py
class Solution:
    def jump(self, nums: List[int]) -> int:
        max_pos = 0
        n = len(nums)

        ans = 0
        end = 0
        for i in range(n-1):
            if i <= max_pos:
                max_pos = max(max_pos, i + nums[i])

            if i == end:
                end = max_pos
                ans += 1
        return ans
```

### 81. 划分字母区间

problem: 给你一个字符串 s 。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。例如，字符串 "ababcc" 能够被分为 ["abab", "cc"]，但类似 ["aba", "bcc"] 或 ["ab", "ab", "cc"] 的划分是非法的。


think: 
同一个字母的第一次出现的下标位置和最后一次出现的下标位置必须出现在同一个片段
使用贪心的思想寻找每个片段可能的最小结束下标

Solution:

```py
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        # 最后一次出现位置
        last_pos = [0 for _ in range(26)]

        for i,c in enumerate(s):
            last_pos[ord(c) - ord('a')] = i

        start = 0
        end = 0
        partition = []
        for i,c in enumerate(s):
            end = max(end, last_pos[ord(c) - ord('a')])

            if i == end:
                partition.append(end - start + 1)
                start = end + 1
        return partition
```

### 82. 单词拆分

problem: 给你一个字符串 s 和一个字符串列表 wordDict 作为字典。如果可以利用字典中出现的一个或多个单词拼接出 s 则返回 true。


think: 
dp[i] 表示 前 i个字符组成的字符 s[0,...i-1]是否能被拆分成字典中若干出现的单词

dp[i] = dp[j] && check(s[j,..i-1])

Solution:

```py
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        
        n = len(s)
        dp = [False for _ in range(n+1)]
        dp[0] = True

        for i in range(1, n+1):
            for j in range(0, i):
                if dp[j] and s[j:i] in wordDict:
                    dp[i] = True
                    break
        
        return dp[n]
```

### 83. 分割等和子集

problem: 给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。


think: 转换成0-1背包问题，选取的数字的和恰好等于整个数组的元素和的一半

dp[i][j] 表示从数组的 [0,i] 下标范围内选取若干个正整数（可以是 0 个），是否存在一种选取方案使得被选取的正整数的和等于 j。

dp[i][0] = True
j >= nums[i] : dp[i][j] = dp[i−1][j] or dp[i−1][j−nums[i]]
j < nums[i] : dp[i][j] = dp[i-1][j]

Solution:

```py
class Solution:
    def canPartition(self, nums: List[int]) -> bool:

        n = len(nums)

        if n < 2:
            return False
        
        sum_num = sum(nums)
        max_num = max(nums)
        target = sum_num // 2
        if sum_num % 2 != 0 or max_num > target:
            return False
        
        dp = [[False] * (target + 1) for _ in range(n)]
        for i in range(n):
            dp[i][0] = True
        dp[0][nums[0]] = True
        for i in range(1, n):
            for j in range(1, target + 1):
                if j >= nums[i]:
                    dp[i][j] = dp[i-1][j] or dp[i-1][j-nums[i]]
                else:
                    dp[i][j] = dp[i-1][j]

        return dp[n-1][target]
```

### 84. 最长公共子序列

problem: 给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。


think:  
dp[i][j] 表示 text1[0:i] text2[0:j] 的最长公共子序列长度

text1[i-1] == text2[j-1] : dp[i][j] = dp[i-1][j-1] + 1
text1[i-1] != text2[j-1] : dp[i][j] = max(dp[i-1][j], dp[i][j-1])

Solution:

```py
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m = len(text1)
        n = len(text2)

        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]

        for i in range(1, m+1):
            for j in range(1, n+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]
```

### 85. 编辑距离
problem: 给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数  。你可以对一个单词进行如下三种操作：
插入一个字符
删除一个字符
替换一个字符


think: 

dp[i][j] 表示 A 前i个字母和B前j个字母之间的编辑距离

dp[i][j-1] + 1

dp[i-1][j] + 1

相同： dp[i-1][j-1]
不同： dp[i-1][j-1] + 1


Solution:

```py
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)

        if m * n == 0:
            return m + n
        
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]

        # 边界
        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j
        

        for i in range(1, m+1):
            for j in range(1, n+1):
                s1 = dp[i-1][j] + 1
                s2 = dp[i][j-1] + 1
                s3 = dp[i-1][j-1]
                if word1[i-1] != word2[j-1]:
                    s3 += 1
                dp[i][j] = min(s1, s2, s3)

        return dp[m][n]
```

### 86. 颜色分类
problem: 给定一个包含红色、白色和蓝色、共 n 个元素的数组 nums ，原地 对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。


think: 
1. 统计出现次数，然后放入数组
2. 使用三个idx, 遍历数组：
若为0, idx_0, idx_1, idx_2 加1
若为1, idx_1, idx_2 加1
若为2, idx_2 加1

Solution:

```py
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # color_cnt = dict()
        # for c in nums:
        #     color_cnt[c] = color_cnt.get(c, 0) + 1
        
        # start = 0
        # for i in [0,1,2]:
        #     if i not in color_cnt:
        #         continue
        #     nums[start: start+color_cnt[i]] = [i] * color_cnt[i]
        #     start = start+color_cnt[i]
        idx0, idx1, idx2 = 0, 0, 0
        for n in nums:

            if n == 0:
                nums[idx2] = 2
                idx2 += 1
                nums[idx1] = 1
                idx1 += 1
                nums[idx0] = 0
                idx0 += 1
                
            elif n == 1:
                nums[idx2] = 2
                idx2 += 1
                nums[idx1] = 1
                idx1 += 1
            else:
                nums[idx2] = 2
                idx2 += 1
        return nums
```

### 87. 下一个排列
problem: 整数数组的一个 排列  就是将其所有成员以序列或线性顺序排列。
例如，arr = [1,2,3] ，以下这些都可以视作 arr 的排列：[1,2,3]、[1,3,2]、[3,1,2]、[2,3,1] 。
整数数组的 下一个排列 是指其整数的下一个字典序更大的排列

think: 
需要将一个左边的「较小数」与一个右边的「较大数」交换，以能够让当前排列变大
这个「较小数」尽量靠右，而「较大数」尽可能小
当交换完成后，「较大数」右边的数需要按照升序重新排列


Solution:

```py
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

```

### 88. 寻找重复数

problem: 给定一个包含 n + 1 个整数的数组 nums ，其数字都在 [1, n] 范围内（包括 1 和 n），可知至少存在一个重复的整数。

think: 

Solution:

```py
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:

        num_cnt = {}
        for n in nums:
            num_cnt[n] = num_cnt.get(n, 0) + 1
            if num_cnt[n] > 1:
                return n
```

### 89. 接雨水

problem: 给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

think: 
维护一个单调栈，存储下标。
从左到右遍历数组，若比栈顶小，直接入栈;
否则 mid = st.pop() 中间出栈， st[-1] 为左边， i为右边
h = min(左边, 右边) - 中间
w = i - st[-1] - 1
s = w * h

Solution:

```py
class Solution:
    def trap(self, height: List[int]) -> int:
        n = len(height)

        res = 0
        st = []
        st.append(0)

        for i in range(1, n):
            while(len(st) > 0 and height[i] > height[st[-1]]):
                mid = st.pop()
                if len(st) > 0:
                    h = min(height[st[-1]], height[i]) - height[mid]
                    w = i - st[-1] -1
                    res += h * w
            st.append(i)
        
        return res
```

### 90. 滑动窗口最大值

problem: 给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。返回 滑动窗口中的最大值 。

think: 
使用大小为k的最大堆保存k个数

Solution:

```py
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        window_heap = []
        for i in range(k):
            heapq.heappush(window_heap, (-nums[i], i))
        ans = [-window_heap[0][0]]
        for i in range(k,n):
            heapq.heappush(window_heap, (-nums[i], i))
            while window_heap[0][1] <= i-k:
                heapq.heappop(window_heap)

            ans.append(-window_heap[0][0])

        return ans
```

### 91. 最小覆盖子串

problem: 给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。

think: 
返回字符串 s 中包含字符串 t 的全部字符的最小窗口
双指针，在 s 上滑动窗口，通过移动 r 指针不断扩张窗口，当窗口包含 t 全部所需的字符后，如果能收缩，我们就收缩窗口直到得到最小窗口。


Solution:

```py
class Solution:
    def minWindow(self, s: str, t: str) -> str:


        need = {}
        for c in t:
            need[c] = need.get(c, 0) + 1

        
        missing = len(t)
        start = 0 
        sub_len = len(s) + 1
        l = 0 # 窗口左端

        for r in range(len(s)):
            c = s[r]
            need[c] = need.get(c, 0)
            if need[c] > 0:
                missing -= 1
            need[c] -= 1

            if not missing:
                while need[s[l]] < 0: # 当前窗口左端字符冗余
                    need[s[l]] += 1
                    l += 1

                if r - l + 1 < sub_len:
                    sub_len = r - l + 1
                    start = l
        
        return "" if sub_len == len(s) + 1 else s[start:start+sub_len]

```

### 92. 缺失的第一个正数

problem: 给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。

think: 
对于一个长度为 N 的数组，其中没有出现的最小正整数只能在 [1,N+1] 中

1. 将数组中所有小于等于 0 的数修改为 N+1
2. 遍历数组中的每一个数 x, 可能已经被打了标记,给数组中的第 ∣x∣−1 个位置的数添加一个负号
3. 在遍历完成之后，如果数组中的每一个数都是负数，那么答案是 N+1，否则答案是第一个正数的位置加 1。

Solution:

```py
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)

        for i in range(n):
            if nums[i] <= 0:
                nums[i] = n + 1
        
        for i in range(n):
            num = abs(nums[i])
            if num <= n:
                nums[num-1] = - abs(nums[num-1])
            
        for i in range(n):
            if nums[i] > 0:
                return i+1

        return n + 1
```

### 93. K 个一组翻转链表

problem: 给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。

think: 
模拟, k个一组
翻转函数： prev, p 先保存 nxt=p.next, p.next = prev, prev = p, p = nxt

拼接： 
tail = prev
nxt = tail.next
head, tail = reverse(head, tail)
prev.next = head
tail.next = nxt
prev = tail
head = tail.next

Solution:

```py
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        def reverse(head, tail):
            prev = tail.next
            p = head

            while prev != tail:
                nxt = p.next
                p.next = prev
                prev = p
                p = nxt
            
            return tail, head

        
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy

        while head:
            tail = prev

            # k个一组
            for i in range(k):
                tail = tail.next
                if not tail:
                    return dummy.next
            
            nxt = tail.next
            
            head, tail = reverse(head, tail)

            # 子链表放回原链表
            prev.next = head
            tail.next = nxt
            prev = tail
            head = tail.next

        return dummy.next
```

### 94. 合并 K 个升序链表

problem: 给你一个链表数组，每个链表都已经按升序排列。请你将所有链表合并到一个升序链表中，返回合并后的链表。

think: 
mege两个有序链表： while ap and bp: tail.next = min(ap, bp)
二分

Solution:

```py

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:

        def mergetwolist(a, b):
            if not a or not b:
                return a if a else b
            
            head = ListNode(0)
            tail = head
            ap = a
            bp = b

            while ap and bp:
                if ap.val < bp.val:
                    tail.next = ap
                    ap = ap.next
                else:
                    tail.next = bp
                    bp = bp.next
                tail = tail.next
            tail.next = ap if ap else bp
            return head.next
        
        def merge(lsts, l, r):
            if l == r:
                return lsts[l]
            if l > r:
                return None
            
            mid = (l+r) // 2
            return mergetwolist(merge(lsts, l, mid), merge(lsts, mid+1, r))
        
        return merge(lists, 0, len(lists) - 1)
```

### 95. 二叉树中的最大路径和

problem: 二叉树中的 路径 被定义为一条节点序列，序列中每对相邻节点之间都存在一条边。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。给你一个二叉树的根节点 root ，返回其 最大路径和 。


think: 
递归求最大左右子树最大路径和（包含 root）
ans = max(ans, left + right - root.val)

Solution:

```py
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        
        ans = -float('inf')
        def dfs(root): #子树最大路径和
            nonlocal ans
            if root == None:
                return 0

            left = max(root.val, dfs(root.left) + root.val)
            right = max(root.val, dfs(root.right) + root.val)

            ans = max(ans, left + right - root.val)

            return max(left, right)
        
        dfs(root)
        return ans
```

### 96. N 皇后

problem: 皇后可以攻击与之处在同一行或同一列或同一斜线上的棋子。n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。

think: 
不同行，不同列：每一行放置一个（不同行），

行作为索引，回溯遍历列

Solution:

```py
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        ans = []

        cb = ['.'*n for _ in range(n)]

        # 回溯
        def backtrack(n, row, cb):
            if row == n:
                ans.append(cb.copy())
                return

            
            for col in range(n):
                if isValid(row, col, cb, n):
                    cb[row] = cb[row][:col] + 'Q' + cb[row][col+1:]
                    backtrack(n, row+1, cb)
                    cb[row] = cb[row][:col] + '.' + cb[row][col+1:]
        
        # 判断棋盘合法： 在row，col位置放置是否合法
        def isValid(row, col, cb, n):

            # 不同列
            for i in range(row):
                if cb[i][col] == 'Q':
                    return False

            # 对角线（45）
            for i, j in zip(range(row-1, -1, -1), range(col+1, n)):
                if cb[i][j] == 'Q':
                    return False

            # 对角线（135）
            for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
                if cb[i][j] == 'Q':
                    return False
            
            return True
        
        backtrack(n, 0, cb)

        return ans
```

### 97. 寻找两个正序数组的中位数

problem: 给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。

think: 

合并两个有序数据，
m+n 为奇数， (m+n)//2
m+n 为偶数, (m+n)//2, (m+n+1)//2, 


Solution:

```py

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        m = len(nums1)
        n = len(nums2)

        i,j = 0, 0
        med1, med2 = 0,0
        k = 0
        ans = []
        while i < m and j < n:
            if nums1[i] < nums2[j]:
                tmp = nums1[i]
                i += 1
            else:
                tmp = nums2[j]
                j += 1

            if k == int((m+n)//2-1):
                med1 = tmp
            elif k == (m+n)//2:
                med2 = tmp

            k += 1
    
        
        while i < m:
            tmp = nums1[i]
            
            if k == int((m+n)//2-1):
                med1 = tmp
            elif k == (m+n)//2:
                med2 = tmp
            k += 1
            i += 1

        
        while j < n:
            tmp = nums2[j]
            
            if k == int((m+n)//2-1):
                med1 = tmp
            elif k == (m+n)//2:
                med2 = tmp
            k += 1
            j += 1
        print(i,j,k,med1, med2)
        return (med1 + med2) / 2 if (m + n) % 2 == 0 else med2

```

### 98. 数据流的中位数

problem: 中位数是有序整数列表中的中间值。如果列表的大小是偶数，则没有中间值，中位数是两个中间值的平均值。

void addNum(int num) 将数据流中的整数 num 添加到数据结构中。
double findMedian() 返回到目前为止所有元素的中位数。与实际答案相差 10-5 以内的答案将被接受。


think: 优先队列，大小堆维护中位数

Solution:

```py
class MedianFinder:

    def __init__(self):
        self.min_heap = []
        self.max_heap = []
        

    def addNum(self, num: int) -> None:
        if not self.min_heap or num <= -self.min_heap[0]:
            heapq.heappush(self.min_heap, -num)

            if len(self.max_heap) + 1 < len(self.min_heap):
                heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))
        else:
            heapq.heappush(self.max_heap, num)
            if len(self.max_heap) > len(self.min_heap):
                heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        

    def findMedian(self) -> float:
        if len(self.min_heap) > len(self.max_heap):
            return -self.min_heap[0]
        
        return (-self.min_heap[0] + self.max_heap[0]) / 2
```

### 99. 柱状图中最大的矩形

problem: 给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。求在该柱状图中，能够勾勒出来的矩形的最大面积。

think: 遍历左右边界，最小高度，更新最大面积


Solution:

```py
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        # n = len(heights)
        # ans = 0
        # for i in range(n):
        #     h_min = float('inf')
        #     for j in range(i,n):
        #         h_min = min(h_min, heights[j])
        #         ans = max(ans, (j-i+1) * h_min)
        # return ans

        stack = []
        heights = [0] + heights + [0]
        res = 0
        for i in range(len(heights)):
            while stack and heights[stack[-1]] > heights[i]:
                tmp = stack.pop()
                res = max(res, (i - stack[-1] - 1) * heights[tmp])
            stack.append(i)
        return res

```

### 100. 最长有效括号

problem: 给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号 子串 的长度。左右括号匹配，即每个左括号都有对应的右括号将其闭合的字符串是格式正确的，比如 "(()())"。

think: 

dp[i] 表示以下标 i 字符结尾的最长有效括号的长度

从前往后遍历字符串 

1. s[i] == ')' and s[i-1] = '('  => dp[i] = dp[i-2] + 2

2. s[i] == ')' and s[i-1] = ')' 如果 s[i−dp[i−1]−1]=‘(’，那么
 => dp[i] = dp[i-1] + dp[i-dp[i-1]-2] + 2


Solution:

```py

class Solution:
    def longestValidParentheses(self, s: str) -> int:
        ans = 0
        n = len(s)
        dp = [0 for _ in range(n)]

        for i in range(1,n):
            if s[i] == ')':
                if s[i-1] == '(':
                    if i >= 2:
                        dp[i] = dp[i-2] + 2
                    else:
                        dp[i] = 2
                elif i - dp[i - 1] > 0 and s[i - dp[i - 1] - 1] == '(':
                    if i - dp[i - 1] >= 2:
                        dp[i] = dp[i - 1] + dp[i - dp[i - 1] - 2] + 2
                    else:
                        dp[i] = dp[i - 1] + 2
                ans = max(ans, dp[i])
        
        return ans

```