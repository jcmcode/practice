#Leetcode problems to practice: 206, 92, 24


def reverse_linked_list(head):
    prev = None
    current = head
    
    while current is not None:
        next = current.next
        current.next = prev
        prev = current 
        current = next
    return prev