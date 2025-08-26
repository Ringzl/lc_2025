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
