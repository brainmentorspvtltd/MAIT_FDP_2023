from datetime import datetime

# Chat Application Using Python

greetIntent = ["hi","hello","hey","hi there","hello there"]
dateIntent = ["date","date please","tell me date","what's the date"]
timeIntent = ["time","time please","tell me time","what's the time"]
chat = True
while chat:
    msg = input("Enter your message : ")
    msg = msg.lower()
    #if msg == "hello" or msg == "hi" or msg == "hey":
    if msg in greetIntent:
        print("Hello User")
    elif msg in dateIntent:
        date = datetime.now().date()
        print("Date is :",date.strftime("%d %b,%y, %a"))
    elif msg in timeIntent:
        time = datetime.now().time()
        print("Time is :",time.strftime("%I:%M:%S, %p"))
    elif msg == "bye":
        print("Bye User")
        chat = False
    else:
        print("I don't understand")
    
