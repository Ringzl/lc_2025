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