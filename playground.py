import os

a = ["{}.pt".format(x) for x in range(10)]

b = sorted(a, reverse=True)[0]

print(b)