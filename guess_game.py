import random

cpu = random.randint(1,100)
counter = 0
game = True
while game:
    user = int(input("Enter your guess : "))
    if cpu == user:
        print("You have guessed the number...")
        game = False
        break
    elif cpu < user:
        print("You have guessed greater number")
    elif cpu > user:
        print("You have guessed smaller number")
    else:
        print("Out of range...")
    counter += 1
    print("Counter :",counter)
    if counter == 5:
        print("You lose the game...Number was",cpu)
        break





