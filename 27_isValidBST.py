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