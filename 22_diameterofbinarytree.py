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
