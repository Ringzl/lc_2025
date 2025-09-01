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