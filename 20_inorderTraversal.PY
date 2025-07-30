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