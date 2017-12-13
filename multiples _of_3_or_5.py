'''
Find the sum of all the multiples of 3 or 5 below 1000.
'''
sum = 0
for temp in range(1,1000):
	if temp % 3 == 0 or temp % 5 == 0:
		sum+= temp
print(sum)
