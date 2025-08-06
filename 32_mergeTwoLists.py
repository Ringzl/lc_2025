class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:

        list3 = ListNode()
        l3 = list3
        while list1 != None and list2 != None:
            if list1.val <= list2.val:
                new_val = list1.val
                list1 = list1.next
            else:
                new_val = list2.val
                list2 = list2.next


            node = ListNode(new_val)
            l3.next = node
            l3 = node
        
        while list1 != None:
            node = ListNode(list1.val)
            l3.next = node
            l3 = node
            list1 = list1.next

        while list2 != None:
            node = ListNode(list2.val)
            l3.next = node
            l3 = node
            list2 = list2.next

        return list3.next