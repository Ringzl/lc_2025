class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:

        def helper(nums, left, right):

            if left > right:
                return 

            # 中间位置/左边作为根节点 向下取整
            mid = left + (right - left) // 2

            root = TreeNode(nums[mid])
            root.left = helper(nums, left, mid - 1)
            root.right = helper(nums, mid + 1, right)

            return root

        return helper(nums, 0, len(nums) - 1)