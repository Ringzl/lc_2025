class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:

        # def check(p, q):
        #     if p == None and q == None:
        #         return True
        #     if p== None or q == None:
        #         return False
            
        #     check_lr = check(p.left, q.right)
        #     check_rl = check(p.right, q.left)
            
        #     if p.val == q.val and check_lr and check_rl:
        #         return True
        #     else:
        #         return False

        # return check(root.left, root.right)

        
        def check(u, v):
            q = deque()
            q.append(u)
            q.append(v)

            while q:
                u = q.popleft()
                v = q.popleft()

                if u == None and v == None:
                    continue

                if u==None or v == None or u.val != v.val:
                    return False
                
                q.append(u.left)
                q.append(v.right)

                q.append(u.right)
                q.append(v.left)

            return True
        
        return check(root, root)
