__author__ = "Katri Leino and Sami Virpioja"
__copyright__ = "Copyright (c) 2020, Aalto Speech Research"
# Python script for human evaluation of recall at 10 metrics.

""" 
PRACTICAL STUFF
Run evaluation test with python human_evaluation_recall10.py
* Put in your user id which is given to you.
* If you want to do test again or stop in the middle, press control+c.
    If you want to continue after the break press control+c. Rename the answer file before starting the new session as the file is rewritten everytime. Combine files before sending them to the organizers.

TEST INSTRUCTIONS
In this test you are to guess a reply to a message. Message and replies are printed to the terminal. Give the number of the answer you think is correct to terminal. 

* note: <MS> means that the user has sent multiple messages. You can ignore it.

"""

import random

# Open and read lines into list.
recall_file = open('eval_topX_recall_at_10_415.csv', 'r')

answers_file = open("answers.txt", "w")

user_answer = input('Käyttäjäkoe ID: ')
answers_file.write("UserID: "+user_answer+"\n")
answers_file.write("set: 4%2.\n")

# Read line by line 
recall_questions = []
with open("eval_topX_recall_at_10_415.csv") as f:
    for i, line in enumerate(f):
        if i == 0: continue # Skip first line

        recall_questions.append(line) 

recall_list_length = len(recall_questions)
recall_indexes = list(range(0, recall_list_length))
random.seed(4)
random.shuffle(recall_indexes)

question_number = 0

# Ignore questions
ignore_list = [279, 269, 235, 23, 221, 233, 255, 129, 247, 231, 7, 177, 119, 123, 163, 239, 5, 263, 9, 125, 219, 291, 237, 15, 215, 11, 115, 159, 3, 259, 169, 36, 256, 80, 48, 234, 0, 278, 206, 210, 220, 86, 218, 102, 296, 230, 170, 270, 18, 124, 214, 106, 280, 56, 298, 138, 122, 38, 216, 146, 252, 272, 74, 58, 82, 70, 162, 254, 268, 118, 204, 208, 44, 120]

# Continuing again after "lopeta" command
# Remember to rename the answer file before starting again and combine by hand afterwards.
#number_of_question_answered = 1
#recall_indexes = recall_indexes[(number_of_question_answered-1)*2:]

random.seed(4)
for idx in recall_indexes:

    if idx%4!=2: continue # Skip odd lines %% %4 !=0 & 2
    if idx in ignore_list: continue # Skip if in ignore list

    question_number += 1

    print('----------------------------------------')
    print('------------------ {:3} -----------------'.format(question_number))
    print('----------------------------------------')
    print()
    question = recall_questions[idx].split('¤')[0]
    print('**Viesti**')
    print()
    print('\n'.join([('>  ' + line.strip()) for line in question.split(' <MS> ')]))
    answers = recall_questions[idx].split('¤')[1].split('|')

    answers_list_length = len(answers)
    answers_indexes = list(range(0, answers_list_length))
    random.shuffle(answers_indexes)

    print()
    print('**Vastausvaihtoehdot**')
    print()

    # Shuffle
    j = 1
    for answer_idx in answers_indexes:
        lines = [('    ' + line.strip()) for line in answers[answer_idx].split(' <MS> ')]
        lines[0] = "{:2}. {}".format(j, lines[0].strip())
        print('\n'.join(lines) + '\n')
        j += 1

    # Print in order
    #for j, answer in enumerate(answers):
    #    print(j+1,".", answer)

    while True:
        user_answer = input("Anna oikean vastauksen numero: ")
        # Stop in the middle
        if user_answer == "Lopeta" or user_answer == "lopeta": 
            break
        try:
            user_answer = int(user_answer)
        except ValueError:
            user_answer = -1
        if 0 < user_answer < len(answers) + 1:
            break
        print('Vastauksen tulee olla numero väliltä 1 - 10')

    # Save 1 if answer correct, 0 if not
    if answers_indexes[int(user_answer)-1] == 0:
        answers_file.write(str(idx)+" 1\n") # correct
    else: 
        answers_file.write(str(idx)+" 0\n") # incorrect

    # Save answer: line index, true=1/false=0
    #answers_file.write(str(idx)+" "+user_answer+"\n")


answers_file.closed
recall_file.closed



