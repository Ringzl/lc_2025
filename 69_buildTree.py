class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        
        # 中序元素位置
        inorder_map = dict()
        for i in range(len(inorder)):
            inorder_map[inorder[i]] = i
        

        # 建树
        # pStart, pEnd: 前序遍历的起始和结束位置
        # iStart, iEnd: 中序遍历的起始和结束位置
        def build(pStart, pEnd, iStart, iEnd):
            if pStart > pEnd or iStart > iEnd:
                return None
            
            # 前序遍历的第一个元素是根节点
            root_val = preorder[pStart]
            root = TreeNode(root_val)

            # 在中序遍历中找到根节点的位置
            root_pos = inorder_map[root_val]

            # 计算左子树的节点数量
            left_num = root_pos - iStart
            root.left = build(pStart+1, pStart+left_num, iStart, root_pos-1)
            root.right = build(pStart+left_num+1, pEnd, root_pos+1, iEnd)
            return root
        
        return build(0, len(preorder)-1, 0, len(inorder)-1)