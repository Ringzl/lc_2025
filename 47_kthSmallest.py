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