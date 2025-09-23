class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:

        def mergetwolist(a, b):
            if not a or not b:
                return a if a else b
            
            head = ListNode(0)
            tail = head
            ap = a
            bp = b

            while ap and bp:
                if ap.val < bp.val:
                    tail.next = ap
                    ap = ap.next
                else:
                    tail.next = bp
                    bp = bp.next
                tail = tail.next
            tail.next = ap if ap else bp
            return head.next
        
        def merge(lsts, l, r):
            if l == r:
                return lsts[l]
            if l > r:
                return None
            
            mid = (l+r) // 2
            return mergetwolist(merge(lsts, l, mid), merge(lsts, mid+1, r))
        
        return merge(lists, 0, len(lists) - 1)