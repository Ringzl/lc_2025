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
