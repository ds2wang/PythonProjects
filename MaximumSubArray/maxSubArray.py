def maxSubArray():
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