import random

cpu = random.randint(1,100)

#Loops - for/while

while True:
    guess = int(input("Guess the number : "))

    if cpu == guess:
        print("You have guessed the number...")
    else:
        print("Your guess is wrong...")

print("Loop Exit...")
