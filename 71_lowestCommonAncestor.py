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