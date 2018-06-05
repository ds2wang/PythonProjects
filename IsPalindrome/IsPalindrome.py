'''
@author: David S Wang
Check if a word is a palindrome
Classic interview question to weed out low level developers 
Included a basic, intuitive way to solve the problem and a more advanced one line solution

Cases not covered by might consider covering in the future:
-sentences with uneven spacing between words
-Inconsistent uppercase/lowercase in word
'''

def isPalindrome(word):
	word = word.lower()
	word = filter(str.isalnum, word)
    for i in range (len(word)/2):
        if word[i] != word[len(word)-i-1]:
            return False
    return True
	
def isPalindromeAdvanced(word):
    return word [:len(word)/2] == word [:len(word)/2:-1] 
