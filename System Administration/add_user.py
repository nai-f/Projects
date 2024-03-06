#! /usr/bin/python3

#Naif Alkaltham

#10/15/2021

import csv
import os
import re

userIDList = []
fullname = []
usernameList = []
officeList = []
phoneList = []
depList =  []
groupList = []


notvalidusersid = []
validusersid = []


def isnumber(value):
    for character in value:
        if character.isdigit():
            return True
    return False

def isname(value):
    for character in value:
        if character.isalpha():
            return True
    return False
os.system("clear")
print("Adding new users to the system.")
print("Please Note: the default password for the new users is ( password )")
print("For testing purposes. Change the password to ( 1$4pizz@ )")



def getall():

    lines = sum(1 for line in open('linux_users.csv'))
    newlines = lines - 1

    with open("linux_users.csv") as file:

        readfile = csv.reader(file, delimiter=',')
        next(readfile)

        for i in range(newlines):
            line = next(readfile)

            #Employee ID 
            uid = line[0]
            userIDList.append(line[0])

            #Employee Username
            try:
                
                if (isname(line[1] or isname(line[2]))):
                    



                    username = (re.sub('[^A-Za-z0-9 ]+', '',(line[2][0] + line[1]))).lower()
                    count = 1
                    while (username in usernameList):
                        username = username + str(count)
                        count += 1
                    #usernameList.append(username)

                    #Employee office
                    if (isnumber(line[3])):
                        ofnumer = line[3]
                        #officeList.append(line[3]) 



                        #Employee phone
                        if (line[4].isdigit()):
                            phnumer = line[4]
                            #phoneList.append(line[4])
                        
                        #Employee Department
                        if (line[5] == "security" or line[5] == "ceo" or line[5] == "office"):
                            depname = line[5]
                            #depList.append(line[5])

                            #Employee Group
                            if (line[6] == "pubsafety" or line[6] == "office"):
                                group = line[6]
                                #groupList.append(line[6])

                                #Valid appened section#
                                #validusersid.append(uid)#
                                validusersid.append(line[0])
                                fullname.append(re.sub('[^A-Za-z0-9 ]+', '',(line[2] + " " + line[1])))
                                usernameList.append(username)
                                officeList.append(line[3])
                                phoneList.append(line[4])
                                depList.append(line[5])
                                groupList.append(line[6])

                            else:
                                notvalidusersid.append(uid)
                                print("Employee ID:  " + userIDList[i] + " is Invalid group")

                            

                        else:
                            notvalidusersid.append(uid)
                            print("Employee ID:  " + userIDList[i] + " is Invalid deparment")
                        
                        
                    else:
                        notvalidusersid.append(uid)
                        print("Employee ID:  " + userIDList[i] + " is Invalid office number")


                else:
                    notvalidusersid.append(uid)
                    print("Employee ID:  " + userIDList[i] + " is Invalid name")


                

            

            except IndexError:
                notvalidusersid.append(uid)
                print ("Employee ID:  " + userIDList[i] + " is Invalid name" )



def creategroup():
    
    groupcreator = set(groupList)
    for i in groupcreator:    
        os.system("sudo groupadd " + i)

def createusers():

    for i in range(len(usernameList)):
        if (groupList[i] != "office"):
            os.system("sudo mkdir -p /home/" + depList[i])
            os.system('sudo useradd -m -d /home/' + depList[i] + "/" + usernameList[i] + ' -g ' + groupList[i] + ' -s /bin/bash' + ' -c ' + '\"' + fullname[i] + '\" ' + usernameList[i])
            os.system("sudo mkhomedir_helper " + usernameList[i])
            print("employee with ID: " + userIDList[i] + " has been add Susccefully")
        else:
            os.system("sudo mkdir -p /home/" + depList[i])
            os.system('sudo useradd -m -d /home/' + depList[i] + "/" + usernameList[i] + ' -g ' + groupList[i] + ' -s /bin/csh' + ' -c ' + '\"' + fullname[i] + '\" ' + usernameList[i])
            os.system("sudo mkhomedir_helper " + usernameList[i])
            print("employee with ID: " + userIDList[i] + " has been add Susccefully")

        os.system("sudo echo password | sudo passwd " + usernameList[i] + " --stdin ")
        os.system("sudo passwd -e " + usernameList[i])
        



getall()

creategroup()
createusers()
