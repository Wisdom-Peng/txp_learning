
class TreeNode:
    def __init__(self,val=0,left=None,right=None):
        self.val ,self.left,self.right = val,left,right

class Solution:
    SEP = ','
    NULL = 'null'

    # 序列化
    def serialize(self,root:TreeNode) -> str:
        def dfs(node):
            if not node:
                vals.append(self.NULL)
                return
            vals.append(str(node.val))
            dfs(node.left)
            dfs(node.right)
        vals = []
        dfs(root)
        return self.SEP.join(vals)
    

    # 反序列化
    def deserialize(self,data:str) -> TreeNode:
        def dfs():
            val = next(it)
            if val == self.NULL:
                return None
            node = TreeNode(int(val))
            node.left = dfs()
            node.right = dfs()
            return node
        it = iter(data.split(self.SEP))
        return dfs()
    
if __name__ == '__main__':
    root = TreeNode(1,TreeNode(2),TreeNode(3,TreeNode(4),TreeNode(5)))
    solution = Solution()
    s = solution.serialize(root)
    print(s)
    s1 = solution.deserialize(s)
    print(s1)

