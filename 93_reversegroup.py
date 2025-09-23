class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        def reverse(head, tail):
            prev = tail.next
            p = head

            while prev != tail:
                nxt = p.next
                p.next = prev
                prev = p
                p = nxt
            
            return tail, head

        
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy

        while head:
            tail = prev

            # k个一组
            for i in range(k):
                tail = tail.next
                if not tail:
                    return dummy.next
            
            nxt = tail.next
            
            head, tail = reverse(head, tail)

            # 子链表放回原链表
            prev.next = head
            tail.next = nxt
            prev = tail
            head = tail.next

        return dummy.next