
num = 11
# loop will start from 2
# loop will execute till num-1
for i in range(2, num):
    if num % i == 0:
        print(f"{num} is not a prime number")
        break
else:
    print(f"{num} is a prime number")
