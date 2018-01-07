'''
Each new term in the Fibonacci sequence is generated by adding the previous two terms. By starting with 1 and 2, the first 10 terms will be:

1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ...

By considering the terms in the Fibonacci sequence whose values do not exceed four million, find the sum of the even-valued terms.
'''

def sumEvenFibinocci():
    help_var = 0
    first = 1 # First Fibinocci Number
    second = 2 # Next Fininocci Number
    while first <= 4000000:
        if first % 2 == 0:
            help_var += second
        first, second = second, first + second
    return str(help_var)


if __name__ == "__main__":
	print(sumEvenFibinocci())
