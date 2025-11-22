#preorder: root -> left -> right
#inorder: left -> root -> right
#postorder: left -> right -> root
#level order: level by level from left to right

#leetcode problems: 257, 230, 124, 107

def preorder_traversal(self, node):
    if node:
        print(node.val, end= '')
        self.preorder_traversal(node.left)
        self.preorder_traversal(node.right)

def inorder_traversal(self, node):
    if node: 
        self.inorder_traversal(node.left)
        print(node.val, end= '')
        self.inorder_traversal(node.right)

def postorder_traversal(self, node):
    if node:
        self.postorder_traversal(node.left)
        self.postorder_traversal(node.right)
        print(node.val, end= '')


