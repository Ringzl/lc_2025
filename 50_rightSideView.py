class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        
        q = deque()
        if root == None:
            return []
        q.appendleft(root)
        ans = []
        while len(q) > 0:
            size = len(q)

            for i in range(size):
                node = q.pop()

                if i == size-1:
                    ans.append(node.val)
                if node.left:
                    q.appendleft(node.left)
                if node.right:
                    q.appendleft(node.right)
        return ans