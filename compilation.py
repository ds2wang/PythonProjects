'''
Created on Feb 26, 2015

@author: DavidS
'''
import sys
from sets import Set
import math
import numpy as np
from _ast import Num
class TreeNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None
'''aaba aabaa''' 
def isPalindrome(word):
    for i in range (len(word)/2):
        if word[i] != word[len(word)-i-1]:
            return False
    return True

#aijjio

def longestPal(word):
    arr = [0]* len(word)
    arr[0] = 1
    for i in range(1, len(word)):
        arr[i] = 1
        if arr[i-1] ==1 and word[i] == word[i-1]:
            arr[i]=2
        if i-arr[i-1]-1>0 and arr[i] == arr[i - arr[i-1]-1]:
            arr[i] = arr[i-1]+2
    longest = ''
    for i in range (len(word)):
        if arr[i]>len(longest):
            longest = word[i-arr[i]+1:i+1]
    print (arr)
    return longest
    
def isPalindromeAdvanced(word):
    return word [:len(word)/2-1] == word [:len(word)/2-1:-1] 

'''
Classic problem of angle between hands of a clock

hour 0 - 11
minute 0-59
seconds 0-59
degree of minute hand is : minute *(1/60) *(360) = minute * 6
hour (Basic): hour *(1/12) * (360) = hour *30 
hour advanced = hour(Basic) + minute/60 * 30

cases:
if(hour > minute) 
 (minute > hour)
 (absoulte difference >180)
'''
def  angleBetweenHands( time):
    hour = int(time[:2])
    minute = int (time[3:])
    return getAngleBetweenHands(hour, minute)

def getAngleBetweenHands(hour, minute):
    hour%=12
    minute%=60
    minuteAngle = minute * 6
    hourAngle = getDegreeHourHand(hour, minute)
    return min(abs(minuteAngle-hourAngle), abs(hourAngle - minuteAngle))

def getDegreeHourHand(hour, minute):
    return int(hour *30 + minute/60.0 * 30)


'''
Was asked this during an interview. Checkout 
http://stackoverflow.com/questions/4606984/maximum-number-of-characters-using-keystrokes-a-ctrla-ctrlc-and-ctrlv
for problem description

my approach:
ctrl a + ctrl c + ctrl v (3 keystrokes) multiplies previous keystrokes by one
ctrl a + ctrl c + ctrl v + ctrl v multiples previous keystrokes by 2
....
etc

so: 
ctrl MaxKeyStroke(n) = max(2*MaxKeyStroke(n-4), 3*MaxKeyStroke(n-4)..... 7 * MaxKeyStroke(n-9))
stop at 7 * MaxKeyStroke(n-9) because:
ctrl 8 * MaxKeyStroke(n-10) < 3* MaxKeyStroke(n-5) = 3 * 3 * MaxKeyStroke(n-10) 

use global list to keep track of values we evaluated so far for MaxKeyStroke
'''
def globalList():
    global MaxKeyStrokeList 
    MaxKeyStrokeList = dict()
    
def MaxKeyStroke(n):
    if(n<8):
        return n
    else:
        if (n in MaxKeyStrokeList.keys()):
            return MaxKeyStrokeList[n]
        listN = list()
        listN.append(2*MaxKeyStroke(n-4))
        listN.append(3*MaxKeyStroke(n-5))
        listN.append(4*MaxKeyStroke(n-6))
        listN.append(5*MaxKeyStroke(n-7))
        listN.append(6*MaxKeyStroke(n-8))
        listN.append(7*MaxKeyStroke(n-9))
        maxN = max(listN)
        MaxKeyStrokeList[n] = maxN
        return maxN
asdf = "asdffff"

'''
p1: get the inputs correctly
read n,
loop n:
    read more inputs
    
p2: correct calculations:
    non contig: just sum the +ve elements -> O(n)
    contig: more tricky -> dp: O(n^2) ?
    
    
tests: that things    


'''
#p1 
def hr1():
    n=int(raw_input())
    for i in range(n):
        numLen = int(raw_input())
        numList = list()
        line = raw_input('')
        for s in line.split(' '):
            num = int(s)
            numList.append(num)
        getSums(numList)
        
        
def getSums(numList):
    maxNum = max(numList)
    sum2 =  getSumNonContig(numList)
    sum1 = getSumContig(numList)
    if sum2==0:
        sum1 = min(maxNum, sum1)
        sum2 = min(maxNum, sum1)
    print(str(sum1)+ ' '+ str(sum2))
    
def getSumNonContig(numList):
    total = 0
    for i in numList:
        if i>0:
            total+=i 
    return total

def getSumContig(numList):
    maxSum = numList[0]
    sumMatrix = [0 for x in range(len(numList))] 
    for i in range(len(numList)):
        curSum = 0
        if i == 0:
            curSum = numList[i]
        else:
            curSum = max(sumMatrix[i-1] + numList[i], numList[i])
        sumMatrix[i] = curSum
        if curSum>maxSum:
            maxSum = curSum           
    return maxSum

def getSumContig2(numList):
    maxSum = numList[0]
    sumMatrix = [[0 for x in range(len(numList))] for x in range(len(numList))] 
    for i in range(len(numList)):
        for j in range(i, len(numList)):
            curSum = 0
            if j == i:
                curSum = numList[j]
            else:
                curSum = sumMatrix[i][j-1] + numList[j]
            sumMatrix[i][j] = curSum
            if curSum>maxSum:
                maxSum = curSum           
    return maxSum
#hr1()
def lcs(a, b):
    
    return []
def fib( n ):
    ans = 0
    fibList = list()
    if len(fibList)>n :
        return fibList[n]
    if n <= 1:
        ans = 1
    else:
        ans = fib(n-1) + fib(n-2) 
    fibList.append(ans)
    return ans
def getEvenSum (num):
    n = 0
    evenSum = 0
    while fib(n) < num:
        if n % 3 == 2:
            evenSum += fib(n)
        n+=1
    print(evenSum)
    return evenSum

def get35Sum(n):
    sum35 = 0
    for i in range(n):
        if i%3 == 0 or i%5 == 0:
            sum35 += i
    
    return sum35

def largestPrimeFactor(n):
    for i in range(2, int(math.sqrt(n))+1):
        if n%i == 0:
            global curMax
            if i>curMax:
                curMax = i
            return max(largestPrimeFactor(n/i), curMax)
    return n
def parse(dictionary, keys):
    val = dictionary
    for key in keys:
        val = val[key]
    return val

IUPAC_TO_BASES = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "R": "AG",
    "Y": "CT",
    "M": "AC",
    "K": "GT",
    "W": "AT",
    "S": "CG",
    "B": "CGT",
    "D": "AGT",
    "H": "ACT",
    "V": "ACG",
    "N": "ACGT"
}

def to_real_dna(bases):
    prevList = ['']
    
    for c in bases: 
        newList = [] 
        possibleBases =IUPAC_TO_BASES[c]
        for base in possibleBases:
            for i in range(len(prevList)):
                newList.append(prevList[i] + base)
                    
        prevList = newList
    return prevList


# you can use print for debugging purposes, e.g.
# print "this is a debug message"

def quasi(A):
    # write your code in Python 2.7
    sortedA = sorted(A)
    ref = sortedA[0]-2
    diff0 = 0 #num of elements same as start of quasi-const subArray
    diff1 = 0 #num elements that differ from start of quasi-const array by 1
    longest = 0 #longest quasi-const
    curLen = 0 #len of current quasi=constdef
    for i in range(len(sortedA)):
        
        #still quasi-const
        if(sortedA[i] - ref<2):
            curLen = curLen+1
            if curLen>longest:
                longest=curLen
            if sortedA[i] - ref == 0:
                diff0=diff0+1
            else:
                diff1=diff1+1
        
        #reset the reference
        else:
            if sortedA[i] - ref == 2 and diff1!=0:
                ref = sortedA[i]-1
                diff0 = diff1
                curLen = diff1 +1
                diff1 = 1
                if curLen > longest:
                    longest = curLen
            else:
                ref = sortedA[i]
                diff0 = 1
                diff1 = 0
                curLen = 1
                if curLen > longest:
                    longest = curLen

    return longest


graph = {
        '1': ['2', '3', '4'],
        '2': ['5', '6'],
        '5': ['9', '10'],
        '4': ['7', '8'],
        '7': ['11', '12']
        }

def bfs(graph, start, end):
    # maintain a queue of paths
    queue = []
    # push the first path into the queue
    queue.append([start])
    while queue:
        # get the first path from the queue
        path = queue.pop(0)
        # get the last node from the path
        node = path[-1]
        # path found
        if node == end:
            return path
        # enumerate all adjacent nodes, construct a new path and push it into the queue
        for adjacent in graph.get(node, []):
            new_path = list(path)
            new_path.append(adjacent)
            queue.append(new_path)

def solution(T):
    # write your code in Python 2.7
    #ok understand question now
    
    #first, build tree
    tree = dict()
    capital = 0
    visited = set([])
    for i in range(len(T)):
        if(T[i] == i):
            capital = i
        else:
            if T[i] not in tree.keys():
                tree[T[i]] = list()
                tree[T[i]].append(i)
            else:

                tree[T[i]].append(i)
            if i not in tree.keys():
                tree[i] = list()
                tree[i].append(T[i])
            else:
                tree[i].append(T[i])
    print(tree)
    prevList = []
    distList = range(len(T))
    prevList.append(capital)
    for i in range (len(T)):
        tempList = []
        for c in prevList:
            distList[i] = distList[i]+ len(set(tree[c]).difference(set(visited)))
            tempList= tempList + Set(tree[c]).difference(Set(visited))
            
        prevList = tempList
            
    print(capital)
    return distList
# Enter your code here. Read input from STDIN. Print output to STDOUT
import fileinput
import sys

# tested on ide
def printColumns():
    tempStr = ''
    for line in sys.stdin:
        tempStr = tempStr + line
    words = tempStr.split()
    words = sorted(words)
    div = len(words)/5
    rem = len(words)%5
    add1 = 0
    if rem>0:
        add1 = 1
    for i in range(div):
        for j in range(5):
            if j<1:
                sys.stdout.write(words[j*div+i]+ ' ')
            elif j<=rem:
                sys.stdout.write(words[j*(div+add1) + i]+ ' ')
            else:
                sys.stdout.write(words[(j-rem)*(div)+ rem*(div+add1)+i]+ ' ')
        print('')
    for j in range(rem):
        sys.stdout.write(words[(j+1)*(div+1)-1]+ ' ')
    print('')
        


    
MAX_SIZE = 2 

linked_list = []
dict1 = {}

class Node:
    value = None
    key = None
    next = None
    prev = None
    
    def __init__(self, key, value, next, prev):
        self.key = key
        self.value = value
    
        


def move_to_head(element):
    element.previous.next = element.next
    linked_list.next = element
    element.next = None
    element.previous = linked_list
    linked_list = element
    # linked_list is the head of the linked_list
    # linked_list.remove(element)
    # linked_list.append(element)
    
def get(str1):
    if str1 not in dict:
        return None
    move_to_head(dict1.get(str1))
    return dict1.get(str1).value
    

def add(str1, val):
    if len(linked_list) == MAX_SIZE:
        node = linked_list.pop(0)
        del dict1[node.key]
        
    e = Node(str1, val)
    dict1[str1] = e
    linked_list.append(e) #goes to head
        
def numeral(roman):
    total = 0
    prevChar = ''
    charMap = {'C':100, 'L': 50, 'X': 10, 'V': 5, 'I':1}
    
    for c in roman:
        total = total + charMap[c]
        if (prevChar!= '' and charMap[prevChar] < charMap[c]):
            total = total - 2*charMap[prevChar] 
        prevChar = c

    return total
# Complete the function below.
import re

def evaluate(expression):
    matchObj5 = re.search( r'(.*)(\+|\-)(.*)$', expression, re.M|re.I)
    matchObj4 = re.search( r'([0-9])(\+|\-)([0-9])$', expression, re.M|re.I)
    matchObj2 = re.search( r'\((.*)\)(\+|\-)([0-9])$', expression, re.M|re.I)
    matchObj3 = re.search( r'([0-9])(\+|\-)\((.*)\)$', expression, re.M|re.I)
    matchObj1 = re.search( r'\((.*)\)(\+|\-)\((.*)\)$', expression, re.M|re.I)
    if matchObj1:
        if matchObj1.group(2) == '+':
            return evaluate(matchObj1.group(1)) + evaluate(matchObj1.group(3))
        else:
            return evaluate(matchObj1.group(1)) - evaluate(matchObj1.group(3))
    elif matchObj2:
        if matchObj2.group(2) == '+':
            return evaluate(matchObj2.group(1)) + int(matchObj2.group(3))
        else:
            return evaluate(matchObj2.group(1)) - int(matchObj2.group(3))
    elif matchObj3:
        if matchObj3.group(2) == '+':
            return int(matchObj3.group(1)) + evaluate(matchObj3.group(3))
        else:
            return int(matchObj3.group(1)) - evaluate(matchObj3.group(3)) 
    elif matchObj4:
        if matchObj4.group(2) == '+':
            return int(matchObj4.group(1)) + int(matchObj4.group(3))
        else:
            return int(matchObj4.group(1)) - int(matchObj4.group(3))
    else:
        if matchObj5.group(2) == '+':
            return int(matchObj5.group(1)) + int(matchObj5.group(3))
        else:
            return int(matchObj5.group(1)) - int(matchObj5.group(3))    

def moveRobot(cmds):
    velos = [[0,1], [1,0], [0,-1], [-1,0]]
    v=0
    pos = [0,0]
    dists=[]
    for i in range(901):
        if i%300 == 1:
            dists.append(calcDist(pos))
        for c in cmds:
            velo = velos[v]
            if c == 'G':
                pos[1] = pos[1]+velo[1]
                pos[0] = pos[0]+velo[0]
            elif c == 'L':
                v= (v-1)%4
            else:
                v=(v+1)%4
    print (dists)
    return not (dists[0]<dists[1] and dists[1]<dists[2] )  
def calcDist(pos):
    return pow(pos[0],2) + pow(pos[1],2)

def fizz(N):
    for i in range(1,N+1):
        word = ''
        if i%3 ==0:
            word = word + 'Fizz'
        if i%5 ==0:
            word = word + 'Buzz'
        if i%7 ==0:
            word = word + 'Woof'
        if word == '':
            word = str(i)
        print (word)

def doStuff(A):
    N = len(A)
    result = 0
    for i in xrange(N):
        for j in xrange(N):
            if A[i] == A[j]:
                result = max(result, abs(i - j))
    return result

def soln2(A): # to replace solution
    N = len(A)
    nums = dict()
    result = 0
    for i in range(N):
        if  nums.has_key(A[i]):
            if abs(i-nums[A[i]])>result:
                result = abs(i-nums[A[i]])
        else:
            nums[A[i]] = i
    return result

A = [6,1,1,1,1,1,6,32,45,6,1,2,7,2,4,71,5,1,7]

def solution3(A): # testing on ide
    N = len(A)
    result = 0
    for i in xrange(N):
        for j in xrange(N):
            if A[i] == A[j]:
                result = max(result, abs(i - j))
    return result

def calcMac(S):
    numStack = []
    for c in S:
        if c<= '9' and c>='0':
            numStack.append(int(c))
        else:
            try:
                num1 = numStack.pop()
                num2 = numStack.pop()
                if c=='+':
                    result = num1 + num2
                else:
                    result = num1 * num2
                if result > '4095': #max value of 12 bit
                    return -1
                numStack.append(result)
            except:
                return -1
    if len(numStack) ==0:
        return -1
    return numStack.pop() 

def anagrams(wordlist):
    anagrams =[]
    wordMap = dict()
    for word in wordlist:
        sortedWord = ''.join(sorted(word))
        if not wordMap.has_key(sortedWord):
            wordMap[sortedWord] = set()
        wordMap[sortedWord].add(word)
    for word in wordlist:
        sortedWord = ''.join(sorted(word))
        if len(wordMap[sortedWord])>1:
            anagrams.append(word)
    return anagrams
    
    
def mergeNodes(list1, list2):
    newList = None #head of new list
    curNode = None
    while(list1!= None and list2!=None):
        newNode = None
        if list1.value>list2.value:
            newNode = Node(list2.value)
            list2 = list2.next
        else:    
            newNode = Node(list1.value)
            list1 = list1.next

        if newList == None:
            curNode = newNode
            newList = newNode
        else:
            curNode.next = newNode
            curNode = newNode

    leftOver = None
    if list1 == None:
        leftOver = list2
    else:
        leftOver = list1
    
    while leftOver !=None:
        newNode = Node(leftOver.value)
        leftOver = leftOver.next
        curNode.next = newNode
        curNode = newNode    

def couldBePalindrome(word):
    charMap = dict()
    for c in word:
        if charMap.has_key(c):
            charMap[c] = charMap[c]+1
        else:
            charMap[c] = 1
    oddChars = set()
    for c in charMap.keySet():
        if not charMap[c]%2 ==0:
            oddChars.add(c)
            
    if len(word)%2==0:
        if len(oddChars) != 0:
            return False
    else: 
        if len(oddChars)!=1:
            return False
    return True


#import IntervalTree?

    

class Interval:
    start = None
    end = None
    def __init__(self, start, end):
        self.start = start
        self.end = end
        
class IntervalTree():
    intervalList = []
    def __init__(self):
        pass
    def getFirst(self):
        return self.intervalList[0]
    def getLast(self):
        return self.intervalList[0]
    def get(self, interval):
        return self.intervalList[0]
    def getNext(self):
        return self.intervalList[0]
    
def getOptimalPath(intervals):
    
    tree = IntervalTree()
    for interval in intervals:
        tree.add(interval)
    
    #determine best path
    first = tree.getFirst()
    last = tree.getLast()
    intervalList = []
    intervalList.add(first)
    curPoint = first
    
    while curPoint.end < last.end:
        
        nextPossibleIntervals = tree.get(curPoint.end)
        if(len(nextPossibleIntervals) !=0):
            for interval in nextPossibleIntervals:
                if interval.end > curPoint.end:
                    curPoint = interval
        else:
            curPoint = tree.getNext(curPoint.end)
            
        intervalList.add(curPoint)
        
    return intervalList

def mergeLists(lists):
    if(len(lists) == 0):
        return None
    elif len(lists)==1:
        return lists[0]
    elif len(lists)==2:
        return merge2Lists(list[0], list[1])
    return merge2Lists(mergeLists(lists[:len(lists)/2]), mergeLists(lists[len(lists)/2:]) )


def merge2Lists(list1, list2):
    newList = list()
    list1Index = 0;
    list2Index = 0;
    while(list1Index<len(list1) and list2Index<len(list2)):
        if list1[list1Index] > list2[list2Index]:
            newList.append(list2[list2Index])
            list2Index = list2Index +1
        else:
            newList.append(list1[list1Index])
            list1Index = list1Index +1
            
    while list1Index<len(list1):
         newList.append(list1[list1Index])
         list1Index = list1Index +1
            
    while list2Index<len(list2):
         newList.append(list2[list2Index])
         list2Index = list2Index +1
                
    return newList