class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:

        # A = headA
        # B = headB

        # while A != B:
        #     A = A.next if A else headB
        #     B = B.next if B else headA

        # return A

        m = 0
        n = 0
        p, q = headA, headB
        while p != None:
            p = p.next
            m += 1
        while q != None:
            q = q.next
            n += 1

        p, q = headA, headB
        if m <= n:
            for i in range(n-m):
                q = q.next
        else:
            for i in range(m-n):
                p = p.next

        while (q != None) and (p != None):
            if p == q:
                return p
            p = p.next
            q = q.next

        return None