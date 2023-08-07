
num = 11
prime = True
# loop will start from 2
# loop will execute till num-1
for i in range(2, num):
    if num % i == 0:
        prime = False
        break

if prime:
    print(f"{num} is a prime number")
else:
    print(f"{num} is not a prime number")
