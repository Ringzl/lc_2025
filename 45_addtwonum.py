class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        
        # s1 = ""
        # while l1 != None:
        #     s1 = str(l1.val) + s1
        #     l1 = l1.next
        
        # s2 = ""
        # while l2 != None:
        #     s2 = str(l2.val) + s2
        #     l2 = l2.next

        # s_sum = reversed(str(int(s1) + int(s2)))

        # res = ListNode()
        # p = res
        # for s in s_sum:
        #     p.next = ListNode(int(s))
        #     p = p.next
        
        # return res.next

        pre = ListNode(0)
        cur = pre
        carry = 0
        while (l1 != None or l2 != None):

            x = l1.val if l1 != None else 0
            y = l2.val if l2 != None else 0

            s = x + y + carry
            carry = s // 10
            s = s % 10
            cur.next = ListNode(s)

            cur = cur.next

            if l1 != None:
                l1 = l1.next
            if l2 != None:
                l2 = l2.next

        if carry != 0:
            cur.next = ListNode(carry)

        return pre.next