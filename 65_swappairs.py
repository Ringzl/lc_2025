class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:


        dummy = ListNode()
        dummy.next = head
        pre = dummy
        while pre.next != None and pre.next.next != None:
            node1 = pre.next
            node2 = pre.next.next

            pre.next = node2
            node1.next = node2.next
            node2.next = node1
            pre = node1

        return dummy.next