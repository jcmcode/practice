#Leetcode Problems to try: 167, 15, 11

def is_palindrome(string):
    start = 0
    end = len (string_ = 1)
    
    while start < end:
        if string[start] != string[end]:
             return False
        start += 1
        end -= 1
         
    return True