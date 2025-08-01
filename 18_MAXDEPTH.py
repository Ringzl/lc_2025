class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:

        if root == None:
            return 0

        else:
            left = self.maxDepth(root.left)
            right = self.maxDepth(root.right)
            return max(left, right) + 1