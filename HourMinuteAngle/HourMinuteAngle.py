'''
Classic problem of angle between hands of a clock

My approach:
hour 0 - 11
minute 0-59
seconds 0-59
degree of minute hand is : minute *(1/60) *(360) = minute * 6
hour (Basic): hour *(1/12) * (360) = hour *30 
hour advanced = hour(Basic) + minute/60 * 30

cases to watch for:
overflowing minutes and/or hour values
negative values
300 degrees = 60 degrees difference,
240 degrees = 120 degrees, etc
'''

def getDegreeDiferenceBetweenClockHands(hour, minute):
    hour %= 12
    minuteAngle = minute * 6
    hourAngle = getDegreeHourHand(hour, minute)
	angle = abs(minuteAngle-hourAngle)
    return min(angle, 360 - angle)

def getDegreeHourHand(hour, minute):
    return int(hour *30 + minute/60.0 * 30)
