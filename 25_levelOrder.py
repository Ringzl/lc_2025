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