import random
import os

import argparse

parse = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parse.add_argument('--file_path', type=str, default="output/",
                    help='Absolute File path for the images to be saved to')

parse.add_argument('--num_of_files', type=int, default=1024,
                    help='Number of different files for the LSTM to train on')

args = parse.parse_args()

#get the current working directory
working_directory = os.getcwd()
#initalise the output string
out_string = ""
#Choice of colour options
colour_options = ['magenta', 'red', 'yellow', 'green']
#Open the file input txt for writing
with open("data/trainFiles/input.txt", "w") as out_file:
    #Create num_of_files versions of the desired output and append them all to a single string
    for i in range(args.num_of_files):
        out_string += "import turtle\n"
        out_string += "import autopy\n"
        out_string += "speed = "+str(random.randint(1,10))+"\n"
        out_string += "pensize = "+str(random.randint(1,5))+"\n"
        out_string += "length = "+str(random.randint(100,450))+"\n"
        out_string += "turtle.speed(speed)\n"
        out_string += "turtle.pensize(pensize)\n"
        out_string += "turtle.color(\""+str(random.choice(colour_options))+"\")\n"
        out_string += "turtle.pendown()\n"
        out_string += "for i in range(4):\n"
        out_string += "\tturtle.forward(length)\n"
        out_string += "\tturtle.left(90)\n"
        out_string += "turtle.hideturtle()\n"
        out_string += "turtle.penup()\n"
        out_string += "bitmap = autopy.bitmap.capture_screen()\n"
        out_string += "bitmap.save('"+working_directory+"/"+args.file_path+"Square"+str(i)+".png')"
        out_string += "\n\n"
    #Write the string to the file
    out_file.write(out_string)
