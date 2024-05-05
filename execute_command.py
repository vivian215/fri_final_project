import recognize_command
import recognize_object
import gesture_recognizer_still as gesture
import sys

arguments = sys.argv[1:]
imagePath = arguments[0]

#listens for command and returns which command it recognized
output_command = recognize_command.recognize_command()

#execute the command
if output_command == 0: #move in direction
    direction = gesture.getFingerDirection(imagePath)
    print("Got it! I am moving to the " + direction + ".")
elif output_command == 1: #detect object
    findNearestObj = recognize_object.recognize_object(imagePath)
    print("The object you are pointing at is a " + findNearestObj() + "!")
else:
    print("I did not recognize that command. Please try again - I can either move in the direction you point or tell you what an object you point to is!")


